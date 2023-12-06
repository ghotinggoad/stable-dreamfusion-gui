import os
import json
import gradio as gr
import sys
import gc
import torch
import cv2
import zipfile
import preprocess_image
import numpy as np
from contextlib import nullcontext
from nerf.utils import *
from guidance import zero123_utils
from omegaconf import OmegaConf
from torchvision import transforms
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange
import argparse
from nerf.provider import NeRFDataset

# main

def main():
    # global variables used to parse variables between functions that are not users' direct input
    global settings
    global current_tab
    global workspace_name
    global single_image
    global max_epoch
    global zero123_weights

    with gr.Blocks(title="stable-diffusion-gui") as app:
        # make temp directory
        os.makedirs("temp", exist_ok=True)
        os.makedirs("workspaces", exist_ok=True)
        
        # initialize global variables and configure gui based on loaded settings
        settings = load_settings()
        current_tab = 3
        if settings["info_tab_on_launch"]:
            current_tab = 0
        workspace_name = "temp"
        single_image = False
        max_epoch = 0
        zero123_weights = [64, 8, 2, 8, 1, 1]

        with gr.Tabs(selected=current_tab) as tabs:
            with gr.Tab(label="new workspace", id=3) as new_workspace_tab:
                # components
                gr.Markdown(
                    """
                    ### enter a workspace name, upload an image and click on preprocess image to start. if no name is entered, the workspace will take the default name "test".
                    """)
                workspace_name_input = gr.Textbox(label="workspace name (no special characters including spaces, only underscores)")
                with gr.Row():
                    image_input = gr.Image(height=512, width=512, label="image")
                remove_background_button = gr.Button(value="remove background", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="secondary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                new_workspace_tab.select(fn=lambda: globals().update(current_tab=3))
                remove_background_button.click(fn=lambda: print(end="")).success(fn=remove_background_button_handler, inputs=[workspace_name_input, image_input], outputs=image_input)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
                
            with gr.Tab(label="six-view generation", id=4) as six_view_generation_tab:
                # components
                gr.Markdown(
                    """
                    ### click on the "generate" button to begin generating the novel images from the different viewpoints.
                    ### use the slider to cycle through the generated images where 0=front, 1=right, 2=back, 3=left, 4=top, 5=bottom.
                    """)
                with gr.Row():
                    image_input = gr.Image(height=512, width=512, interactive=False)
                    images_viewer_output = gr.Image(label="image")
                images_viewer_slider_input = gr.Slider(minimum=0, maximum=5, label="slide to view images generated from different angles", step=1, interactive=False)
                with gr.Row():
                    single_image_input = gr.Checkbox(value=False, label="single image only")
                    boosted_weights_input = gr.Checkbox(value=False, label="boosted weights (model will more closely follow the generated images of the sides and back)")
                generate_button = gr.Button(value="generate", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                six_view_generation_tab.select(fn=lambda: globals().update(current_tab=4)).success(fn=return_input_image_handler, outputs=image_input)
                single_image_input.select(fn=single_image_checkbox_handler, inputs=single_image_input, outputs=images_viewer_output)
                boosted_weights_input.select(fn=boosted_weights_checkbox_handler, inputs=boosted_weights_input)
                generate_button.click(fn=lambda: print(end="")).success(fn=six_view_generation_handler, outputs=[images_viewer_output, images_viewer_slider_input])
                images_viewer_slider_input.change(fn=images_viewer_slider_handler, inputs=images_viewer_slider_input, outputs=images_viewer_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
            
            with gr.Tab(label="model generation", id=5) as model_generation_tab:
                # components
                gr.Markdown(
                    """
                    ### after you have finished inputting the desired parameters, click on the "generate" button to begin generating the novel images from the different viewpoints.
                    ### the model will be shown on the right image window and the mesh fill be available to download right above the "generate" button.
                    """)
                with gr.Row():
                    image_input = gr.Image(height=512, width=512, interactive=False)
                    images_viewer_output = gr.Image(label="image")
                images_viewer_slider_input = gr.Slider(minimum=0, maximum=0, label="slide to view generated model from different angles", step=1, interactive=False)
                with gr.Row():
                    random_seed_input = gr.Checkbox(value=True, label="random seed")
                    seed_input = gr.Number(value=None, label="seed", precision=0)
                    size_input = gr.Number(value=64, label="size (n^2, 64 really recommended.)", minimum=64, precision=0, step=1) #64
                with gr.Row():
                    iters_input = gr.Number(value=5000, label="iters (iterations)", precision=0, minimum=1, step=1) #5000
                    lr_input = gr.Number(value=1e-3, label="lr (learning rate)", minimum=1e-5) #1e-3
                    batch_size_input = gr.Number(value=1, label="batch_size", precision=0, minimum=1, step=1) #1
                with gr.Row():
                    dataset_size_train_input = gr.Number(value=100, label="dataset_size_train", precision=0, minimum=1, step=1) #100
                    dataset_size_valid_input = gr.Number(value=8, label="dataset_size_valid", precision=0, minimum=1, step=1) #8
                    dataset_size_test_input = gr.Number(value=100, label="dataset_size_test", precision=0, minimum=1, step=1) #100
                file_output = gr.File(visible=False)
                generate_button = gr.Button(value="generate", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="primary")
                # events
                model_generation_tab.select(fn=lambda: globals().update(current_tab=5)).success(fn=return_input_image_handler, outputs=image_input)
                generate_button.click(fn=lambda: print(end="")).success(fn=model_generation_handler, inputs=[random_seed_input, seed_input, size_input, iters_input, lr_input, batch_size_input, dataset_size_train_input, dataset_size_valid_input, dataset_size_test_input], outputs=[images_viewer_output, images_viewer_slider_input, file_output])
                images_viewer_slider_input.change(fn=images_viewer_slider_handler, inputs=images_viewer_slider_input, outputs=images_viewer_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
            
            with gr.Tab(label="model finetuning", id=6) as model_fine_tuning_tab:
                # components
                gr.Markdown(
                    """
                    ### after you have finished inputting the desired parameters, click on the "finetune" button to begin generating the novel images from the different viewpoints.
                    ### the model will be shown on the right image window and the mesh fill be available to download right above the "finetune" button.
                    """)
                with gr.Row():
                    image_input = gr.Image(height=512, width=512, interactive=False)
                    images_viewer_output = gr.Image(label="image")
                images_viewer_slider_input = gr.Slider(minimum=0, maximum=0, label="slide to view generated model from different angles", step=1, interactive=False)
                with gr.Row():
                    random_seed_input = gr.Checkbox(value=True, label="random seed")
                    seed_input = gr.Number(value=None, label="seed", precision=0)
                with gr.Row():
                    size_input = gr.Number(value=64, label="size (n^2, 64 really recommended.)", minimum=64, precision=0, step=1) #64
                    tet_grid_size_input = gr.Dropdown(label="tet_grid_size", choices=["32", "64", "128", "256"], value="128")
                with gr.Row():
                    iters_input = gr.Number(value=5000, label="iters (iterations)", precision=0, minimum=1, step=1) #5000
                    lr_input = gr.Number(value=1e-3, label="lr (learning rate)", minimum=1e-5) #1e-3
                    batch_size_input = gr.Number(value=1, label="batch_size", precision=0, minimum=1, step=1) #1
                with gr.Row():
                    dataset_size_train_input = gr.Number(value=100, label="dataset_size_train", precision=0, minimum=1, step=1) #100
                    dataset_size_valid_input = gr.Number(value=8, label="dataset_size_valid", precision=0, minimum=1, step=1) #8
                    dataset_size_test_input = gr.Number(value=100, label="dataset_size_test", precision=0, minimum=1, step=1) #100
                file_output = gr.File(visible=False)
                finetune_button = gr.Button(value="finetune", variant="primary")
                with gr.Row():
                    previous_tab_button = gr.Button(value="previous", variant="primary")
                    next_tab_button = gr.Button(value="next", variant="secondary")
                # events
                model_fine_tuning_tab.select(fn=lambda: globals().update(current_tab=6)).success(fn=return_input_image_handler, outputs=image_input)
                finetune_button.click(fn=lambda: print(end="")).success(fn=model_finetuning_handler, inputs=[random_seed_input, seed_input, size_input, tet_grid_size_input, iters_input, lr_input, batch_size_input, dataset_size_train_input, dataset_size_valid_input, dataset_size_test_input], outputs=[images_viewer_output, images_viewer_slider_input, file_output])
                images_viewer_slider_input.change(fn=images_viewer_slider_handler, inputs=images_viewer_slider_input, outputs=images_viewer_output)
                previous_tab_button.click(fn=previous_tab_button_handler, outputs=tabs)
                next_tab_button.click(fn=next_tab_button_handler, outputs=tabs)
            
            with gr.Tab(label="workspace manager", id=2) as file_manager_tab:
                # components
                gr.Markdown(
                    """
                    ### this tab is used to view, download and delete workspaces, simply select a workspace name to start.
                    """
                )
                with gr.Row():
                    images_viewer_output = gr.Image(label="model viewer", interactive=False)
                    with gr.Column():
                        images_viewer_slider_input = gr.Slider(minimum=0, maximum=0, label="slide to change viewpoint of model", step=1)
                        workspace_name_input = gr.Dropdown(choices=os.listdir("workspaces"), label="workspace name")
                        finetune_button = gr.Button(visible=False)
                        delete_button = gr.Button(visible=False)
                        file_output = gr.File(visible=False)
                # events
                file_manager_tab.select(fn=lambda: globals().update(current_tab=2)).success(fn=lambda: gr.Dropdown(choices=os.listdir("workspaces"), label="workspace name"), outputs=workspace_name_input)
                images_viewer_slider_input.change(fn=workspace_manager_name_handler, inputs=workspace_name_input).success(fn=images_viewer_slider_handler, inputs=images_viewer_slider_input, outputs=images_viewer_output)
                workspace_name_input.input(fn=load_workspace_handler, inputs=workspace_name_input, outputs=[images_viewer_output, images_viewer_slider_input, finetune_button, delete_button, file_output])
                finetune_button.click(fn=finetune_workspace_handler, inputs=workspace_name_input, outputs=[tabs])
                delete_button.click(fn=delete_workspace_handler, inputs=workspace_name_input, outputs=[images_viewer_output, images_viewer_slider_input, workspace_name_input, delete_button, file_output])
                
            with gr.Tab(label="settings", id=1) as settings_tab:
                # components
                gr.Markdown(
                    """
                    ### click on the "save settings" button to after you have dialed in your settings, otherwise the settings will not be updated (and saved).
                    """)
                info_tab_on_launch = gr.Checkbox(value=settings["info_tab_on_launch"], label="load up info tab on launch")
                zero123_checkpoint_input = gr.Dropdown(choices=["zero123-xl", "105000", "165000"], value=settings["zero123_checkpoint"], label="zero123 model checkpoint")
                backbone_input = gr.Dropdown(choices=["grid", "vanilla", "grid_tcnn", "grid_taichi"], value=settings["backbone"], label="nerf backbone")
                optimizer_input = gr.Dropdown(choices=["adan", "adam"], value=settings["optimizer"], label="optimizer")
                fp16_input = gr.Checkbox(value=settings["fp16"], label="use float16 instead of float32 for training")
                save_button = gr.Button(value="save settings", variant="primary")
                # events
                settings_tab.select(fn=lambda: globals().update(current_tab=1))
                save_button.click(fn=save_settings_handler, inputs=[info_tab_on_launch, backbone_input, optimizer_input, fp16_input, zero123_checkpoint_input], outputs=save_button)
                    
            with gr.Tab(label="info", id=0) as info_tab:
                # components
                gr.Markdown(
                    """
                    # image to 3d model generation
                    a final year workspace by oh zhi hua (rod) for nanyang technological university computer engineering program.
                    
                    ## Introduction
                    Welcome to stable-dreamfusion-gui, a workspace which provides a graphical user interface to generate 3D models from a single image by wrapping the stable-dreamfusion with gradio.
                    
                    The workspace gives the user the option to generate 6 novel viewpoints from the front, back, left, right, top and bottom of the object in the input image, before sending them into stable-dreamfusion for 3D generation.
                    
                    To start, simply click the tab labeled "new workspace" to start exploring.
                    
                    Have fun!
                    
                    ## Tabs
                    new workspace  → start generating 3D models
                    
                    file manager   → manage the existing workspaces, view/finetune/delete workspaces
                    
                    settings       → configure the NeRF backend and default tab when starting the application
                    
                    ## Support
                    If you need support, please submit an issue at "https://github.com/ghotinggoad/stable-dreamfusion-gui/issues"!
                    """)
                # events
                info_tab.select(fn=lambda: globals().update(current_tab=0))
    
    _, local_url, public_url = app.queue(max_size=1).launch(share=True)
    print(local_url)
    print(public_url)
    
    # rmdir temp folder
    delete_directory("temp")
    delete_directory("workspaces/temp")
    # clear ram (including vram)
    clear_memory()

# functions used to interact with system

def clear_memory():
    # called after deleting the items in python
    gc.collect()
    torch.cuda.empty_cache()

def delete_directory(path):
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        os.rmdir(path)
    except:
        print("cannot delete the folder "+path)

def load_settings():
    try:
        with open("settings.json", "r") as file:
            data = json.load(file)
        return data
    except:
        print("settings failed to load")

def save_settings_handler(info_tab_on_launch, backbone, optimizer, fp16, zero123_checkpoint):
    try:
        with open("settings.json", "w") as file:
            settings["info_tab_on_launch"] = info_tab_on_launch
            settings["backbone"] = backbone
            settings["optimizer"] = optimizer
            settings["fp16"] = fp16
            settings["zero123_checkpoint"] = zero123_checkpoint
            
            json.dump(settings, file, indent=4)
            return gr.Button(value="settings saved", variant="primary")
    except:
        print("settings failed to save")

def load_workspace_handler(workspace_name_input):
    global max_epoch
    
    os.makedirs("temp/{}".format(workspace_name_input), exist_ok=True)
    temp = cv2.imread("workspaces/{}/images/image_0.png".format(workspace_name_input), cv2.IMREAD_UNCHANGED)
    cv2.imwrite("temp/{}/image.png".format(workspace_name_input), temp)
    del temp
    
    with open("workspaces/{}/info.json".format(workspace_name_input), "r") as f:
        temp = json.load(f)
    max_epoch = temp["max_epoch"]
    dmtet = temp["dmtet"]
    dataset_size_train = temp["dataset_size_train"]
        
    del temp
    
    os.makedirs("temp/{}/workspace_manager".format(workspace_name_input), exist_ok=True)
    video = cv2.VideoCapture("workspaces/{}/results/df_ep{:04d}_rgb.mp4".format(workspace_name_input, max_epoch))
    image = video.read()[1]
    cv2.imwrite("temp/{}/workspace_manager/df_ep{:04d}_{:04d}_rgb.png".format(workspace_name_input, max_epoch, 0), image)
    for i in range(1, dataset_size_train):
        temp = video.read()[1]
        cv2.imwrite("temp/{}/workspace_manager/df_ep{:04d}_{:04d}_rgb.png".format(workspace_name_input, max_epoch, i), temp)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    with zipfile.ZipFile("temp/{}/{}.zip".format(workspace_name_input, workspace_name_input), "w") as file:
        file.write("workspaces/{}/mesh/albedo.png".format(workspace_name_input), arcname="albedo.png")
        file.write("workspaces/{}/mesh/mesh.mtl".format(workspace_name_input), arcname="mesh.mtl")
        file.write("workspaces/{}/mesh/mesh.obj".format(workspace_name_input), arcname="mesh.obj")
    
    if dmtet:
        return image, gr.Slider(label="slide to change viewpoint of model", minimum=0, maximum=dataset_size_train-1, value=0, step=1), \
            gr.Button(visible=False), gr.Button(value="delete", visible=True, variant="stop"), \
            gr.File(value="temp/{}/{}.zip".format(workspace_name_input, workspace_name_input), label="download", visible=True)
    else:
        return image, gr.Slider(label="slide to change viewpoint of model", minimum=0, maximum=dataset_size_train-1, value=0, step=1), \
            gr.Button(value="finetune", visible=True, variant="primary"), gr.Button(value="delete", visible=True, variant="stop"), \
            gr.File(value="temp/{}/{}.zip".format(workspace_name_input, workspace_name_input), label="download", visible=True)

def finetune_workspace_handler(workspace_name_input):
    global current_tab
    global workspace_name
    global single_image
    current_tab = 6
    workspace_name = workspace_name_input
    
    with open("workspaces/{}/info.json".format(workspace_name_input), "r") as f:
        temp = json.load(f)
    single_image = temp["single_image"]
    
    os.makedirs("temp/{}/six_view_generation".format(workspace_name), exist_ok=True)
    image = cv2.imread("workspaces/{}/images/image_0.png".format(workspace_name_input), cv2.IMREAD_UNCHANGED)
    cv2.imwrite("temp/{}/six_view_generation/image_0.png".format(workspace_name_input), image)
    if not single_image:
        for i in range(1, 6):
            image = cv2.imread("workspaces/{}/images/image_{:1}.png".format(workspace_name_input, i), cv2.IMREAD_UNCHANGED)
            cv2.imwrite("temp/{}/six_view_generation/image_{:1}.png".format(workspace_name_input, i), image)
        
    return gr.Tab(selected=current_tab)

def delete_workspace_handler(workspace_name_input):
    delete_directory("workspaces/{}".format(workspace_name_input))
    return gr.Image(value=None, interactive=False), gr.Slider(minimum=0, maximum=0, value=None, step=1, label="slide to change viewpoint of model"), \
           gr.Dropdown(choices=os.listdir("workspaces"), value=None, label="workspace name"), gr.Button(visible=False), gr.File(visible=False)
                        

# gradio event functions

def previous_tab_button_handler():
    global current_tab
    if current_tab > 3:
        current_tab -= 1
    return gr.Tabs(selected=current_tab)

def next_tab_button_handler():
    global current_tab
    if current_tab < 6:
        current_tab += 1
    return gr.Tabs(selected=current_tab)

def return_input_image_handler():
    image = cv2.cvtColor(cv2.imread("temp/{}/image.png".format(workspace_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    return image

def images_viewer_slider_handler(slider):
    # updates the image based on the slider value, usually to select the "angle" (index of the image)
    global current_tab
    global workspace_name
    global max_epoch
    if current_tab == 2:
        with open("workspaces/{}/info.json".format(workspace_name), "r") as f:
            temp = json.load(f)
        max_epoch = temp["max_epoch"]
        image = cv2.imread("temp/{}/workspace_manager/df_ep{:04d}_{:04d}_rgb.png".format(workspace_name, max_epoch, slider))
    elif current_tab == 4:
        image = cv2.imread("temp/{}/six_view_generation/image_{:1}.png".format(workspace_name, slider))
    elif current_tab == 5:
        image = cv2.imread("temp/{}/workspace_manager/df_ep{:04d}_{:04d}_rgb.png".format(workspace_name, max_epoch, slider))
    elif current_tab == 6:
        image = cv2.imread("temp/{}_dmtet/workspace_manager/df_ep{:04d}_{:04d}_rgb.png".format(workspace_name, max_epoch, slider))
    else:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def single_image_checkbox_handler(single_image_input):
    global current_tab
    global workspace_name
    global single_image
    
    current_tab = 4
    single_image = single_image_input
    
    image = cv2.cvtColor(cv2.imread("temp/{}/image.png".format(workspace_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGB)
    os.makedirs("temp/{}/six_view_generation".format(workspace_name), exist_ok=True)
    
    image = cv2.resize(image, (512, 512))
    cv2.imwrite("temp/{}/six_view_generation/image_0.png".format(workspace_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return image

def boosted_weights_checkbox_handler(boosted_weights_input):
    global current_tab
    global zero123_weights
    
    current_tab = 4
    if boosted_weights_input:
        zero123_weights = [8, 4, 2, 4, 1, 1]
    else:
        zero123_weights = [64, 8, 2, 8, 1, 1]

def workspace_manager_name_handler(workspace_name_input):
    global workspace_name
    workspace_name = workspace_name_input

# zero123/stable-dreamfusion/cv functions

def preprocess(image, recenter=True, size=256, border_ratio=0.2):
    # this checks to if original image is RGBA or RGB then convert it into CV2 format
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # removes the background, keeps only the subject
    carved_image = preprocess_image.BackgroundRemoval()(image) # [H, W, 4]
    mask = carved_image[..., -1] > 0
    
    # predict depth
    dpt_depth_model = preprocess_image.DPT(task='depth')
    depth = dpt_depth_model(image)[0]
    depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-9)
    depth[~mask] = 0
    depth = (depth * 255).astype(np.uint8)
    del dpt_depth_model
    
    # predict normal
    dpt_normal_model = preprocess_image.DPT(task='normal')
    normal = dpt_normal_model(image)[0]
    normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
    normal[~mask] = 0
    del dpt_normal_model
    
    # recenter
    if recenter:
        final_rgba = np.zeros((size, size, 4), dtype=np.uint8)
        final_depth = np.zeros((size, size), dtype=np.uint8)
        final_normal = np.zeros((size, size, 3), dtype=np.uint8)
        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (size - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(depth[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(normal[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    else:
        final_rgba = carved_image
        final_depth = depth
        final_normal = normal
    
    # final_normal = cv2.cvtColor(final_normal, cv2.COLOR_RGB2BGR) # this is only for display onto gradio
    
    return final_rgba, final_depth, final_normal

@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta, xs, ys):
    # taken from zero123/gradio_new.py
    precision_scope = torch.autocast if precision == 'autocast' else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            n_samples = n_samples * len(xs)
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = []
            for x, y in zip(xs, ys):
                T.append([math.radians(x), math.sin(math.radians(y)), math.cos(math.radians(y)), 0])
            T = torch.tensor(np.array(T))[:, None, :].float().to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im)).mode().detach().repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None
            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=cond, batch_size=n_samples, shape=shape, verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=ddim_eta, x_T=None)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()

def generate_novel_views(image, iters, polars, azimuths, size=256):
    global settings
    # polars, top = -90, straight = 0, bottom = 90
    # azimuth, left = -90, front = 0, right = 90, behind = 180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = zero123_utils.load_model_from_config(OmegaConf.load("pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml"), "pretrained/zero123/zero123-xl.ckpt", device)
    model = zero123_utils.load_model_from_config(OmegaConf.load("pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml"), "pretrained/zero123/{}.ckpt".format(settings["zero123_checkpoint"]), device)
    model.use_ema = False
    
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    image = image * 2 - 1
    image = transforms.functional.resize(image, [size, size], antialias=True)
    
    sampler = DDIMSampler(model)
    
    images = []
    x_samples_ddim = sample_model(image, model, sampler, "fp16", size, size, iters, 1, 3.0, 1.0, polars, azimuths)
    for x_sample in x_samples_ddim:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        images.append(np.asarray(x_sample.astype(np.uint8)))
    
    # for i in range(len(polars)):
    #     x_samples_ddim = sample_model(image, model, sampler, "fp32", size, size, iters, 1, 3.0, 1.0, polars[i], azimuths[i], radii[i])
    #     novel_image = []
    #     for x_sample in x_samples_ddim:
    #         x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
    #         novel_image.append(np.asarray(x_sample.astype(np.uint8)))
    #         images.append(novel_image[0])
    #     del x_samples_ddim
    #     del novel_image
    #     clear_memory()
    del model
    del image
    del device
    del sampler
    del x_samples_ddim
    clear_memory()
    
    return images

def generate_model(opt):
    global max_epoch

    if opt.dmtet:    
        opt.h = int(opt.h * opt.dmtet_reso_scale)
        opt.w = int(opt.w * opt.dmtet_reso_scale)
        opt.known_view_scale = 1
        if not opt.dont_override_stuff:            
            opt.t_range = [0.02, 0.50] # ref: magic3D
        if opt.images is not None:
            opt.lambda_normal = 0
            opt.lambda_depth = 0
            if opt.text is not None and not opt.dont_override_stuff:
                opt.t_range = [0.20, 0.50]
        # assume finetuning
        opt.latent_iter_ratio = 0
        opt.albedo_iter_ratio = 0
        opt.progressive_view = False
        # opt.progressive_level = False
    
    # this part of the code loads up the relevant modules to allow the trainer to run (taken from main.py)    
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'grid_tcnn':
        from nerf.network_grid_tcnn import NeRFNetwork
    elif opt.backbone == 'grid_taichi':
        opt.cuda_ray = False
        opt.taichi_ray = True
        import taichi as ti
        from nerf.network_grid_taichi import NeRFNetwork
        taichi_half2_opt = True
        taichi_init_args = {"arch": ti.cuda, "device_memory_GB": 4.0}
        if taichi_half2_opt:
            taichi_init_args["half2_vectorization"] = True
        ti.init(**taichi_init_args)

    if opt.seed is not None:
        seed_everything(int(opt.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRFNetwork(opt).to(device)

    if opt.dmtet and opt.init_with != '':
        if opt.init_with.endswith('.pth'):
            # load pretrained weights to init dmtet
            state_dict = torch.load(opt.init_with, map_location=device)
            model.load_state_dict(state_dict['model'], strict=False)
            if opt.cuda_ray:
                model.mean_density = state_dict['mean_density']
            model.init_tet()
        else:
            # assume a mesh to init dmtet (experimental, not working well now!)
            import trimesh
            mesh = trimesh.load(opt.init_with, force='mesh', skip_material=True, process=False)
            model.init_tet(mesh=mesh)

    train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=opt.dataset_size_train * opt.batch_size).dataloader()

    if opt.optim == 'adan':
        from optimizer import Adan
        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else: # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
    else:
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    guidance = nn.ModuleDict()
    guidance['zero123'] = zero123_utils.Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)
    trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, scheduler_update_every_step=True)
    trainer.default_view_data = train_loader._data.get_default_view_data()
    valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=opt.dataset_size_valid).dataloader(batch_size=1)
    test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=opt.dataset_size_test).dataloader(batch_size=1)
    
    # save max_epoch
    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    
    # starts the training
    trainer.train(train_loader, valid_loader, test_loader, max_epoch)
    
    # saves mesh
    trainer.save_mesh()

def remove_background_button_handler(workspace_name_input, image_input):
    global current_tab
    global workspace_name
    
    current_tab = 3
    
    workspace_name = workspace_name_input
    if image_input.shape[-1] == 4:
        image_input = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
    
    height, width, channels = image_input.shape
    size = max(height, width)
    image = np.zeros((size, size, channels), dtype=np.uint8)
    x_offset = (size-width)//2
    y_offset = (size-height)//2
    image[y_offset:y_offset + height, x_offset:x_offset + width] = image_input
    
    image = preprocess_image.BackgroundRemoval()(image)
    image = cv2.resize(image, (512, 512))
    
    # image = preprocess(image_input, size=512)[0]
    os.makedirs("temp/{}".format(workspace_name), exist_ok=True)
    cv2.imwrite("temp/{}/image.png".format(workspace_name), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite("workspaces/{}/image.png".format(workspace_name), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    return image

def six_view_generation_handler():
    global current_tab
    global workspace_name
    
    current_tab = 4
    
    image = cv2.cvtColor(cv2.imread("temp/{}/image.png".format(workspace_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGB)
    os.makedirs("temp/{}/six_view_generation".format(workspace_name), exist_ok=True)
    
    polars = [0.0, 0.0, 0.0, -90.0, 90.0]
    azimuths = [90.0, 180.0, -90.0, 0.0, 0.0]
    images = generate_novel_views(image, 200, polars, azimuths)
    
    image = cv2.resize(image, (512, 512))
    cv2.imwrite("temp/{}/six_view_generation/image_0.png".format(workspace_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    for i in range(5):
        images[i] = cv2.resize(images[i], (512, 512))
        cv2.imwrite("temp/{}/six_view_generation/image_{:1}.png".format(workspace_name, i+1), cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
    
    return image, gr.Slider(minimum=0, maximum=5, label="slide to view images generated from different angles", step=1, interactive=True)

def model_generation_handler(random_seed, seed, size, iters, lr, batch_size, dataset_size_train, dataset_size_valid, dataset_size_test):
    global settings
    global current_tab
    global workspace_name
    global single_image
    global zero123_weights
    
    current_tab = 5
    
    # load opt from opt.json to parse into the trainer object in util.py
    opt = {}
    with open("opt.json", "r") as f:
        opt = json.load(f)
        
    # these arguments are for zero123
    opt["backbone"] = settings["backbone"]
    opt["optim"] = settings["optimizer"]
    opt["fp16"] = settings["fp16"]
    opt["workspace"] = "workspaces/{}".format(workspace_name)
    if random_seed:
        opt["seed"] = None
    else:
        opt["seed"] = seed
    opt["h"] = size
    opt["w"] = size
    opt["iters"] = iters
    opt["lr"] = lr
    opt["batch_size"] = batch_size
    opt["dataset_size_train"] = dataset_size_train
    opt["dataset_size_valid"] = dataset_size_valid
    opt["dataset_size_test"] = dataset_size_test
    opt["exp_start_iter"] = opt["exp_start_iter"] or 0
    opt["exp_end_iter"] = opt["exp_end_iter"] or opt["iters"]
    if single_image:
        opt["images"] = ["temp/{}/model_generation/image_0_rgba.png".format(workspace_name)]
        opt["ref_polars"] = [90.0]
        opt["ref_azimuths"] = [0.0]
        opt["ref_radii"] = [3.2]
        opt["zero123_ws"] = [1]
    else: 
        opt["images"] = ["temp/{}/model_generation/image_0_rgba.png".format(workspace_name), "temp/{}/model_generation/image_1_rgba.png".format(workspace_name), "temp/{}/model_generation/image_2_rgba.png".format(workspace_name), "temp/{}/model_generation/image_3_rgba.png".format(workspace_name), "temp/{}/model_generation/image_4_rgba.png".format(workspace_name), "temp/{}/model_generation/image_5_rgba.png".format(workspace_name)]
        opt["ref_polars"] = [90.0, 90.0, 90.0, 90.0, 180.0, 0.0001]
        opt["ref_azimuths"] = [0.0, 90.0, 180.0, -90.0, 0.0, 0.0]
        opt["ref_radii"] = [3.2, 3.2, 3.2, 3.2, 3.2, 3.2]
        opt["zero123_ws"] = zero123_weights

    opt = argparse.Namespace(**opt)
    
    # preprocess and save images to temporary folder in order to make calls to stable-dreamfusion trainer without making changes to its code
    os.makedirs("temp/{}/model_generation".format(workspace_name), exist_ok=True)
    
    image = cv2.cvtColor(cv2.imread("temp/{}/six_view_generation/image_0.png".format(workspace_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    image_rgba, image_depth, image_normal = preprocess(image, size=1024)
    cv2.imwrite("temp/{}/model_generation/image_0_rgba.png".format(workspace_name), cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite("temp/{}/model_generation/image_0_depth.png".format(workspace_name), image_depth)
    cv2.imwrite("temp/{}/model_generation/image_0_normal.png".format(workspace_name), image_normal)
    del image_rgba, image_depth, image_normal
    if not single_image:    
        for i in range(1, 6):
            image = cv2.cvtColor(cv2.imread("temp/{}/six_view_generation/image_{:1}.png".format(workspace_name, i), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            image_rgba, image_depth, image_normal = preprocess(image, size=1024)
            cv2.imwrite("temp/{}/model_generation/image_{:1}_rgba.png".format(workspace_name, i), cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
            cv2.imwrite("temp/{}/model_generation/image_{:1}_depth.png".format(workspace_name, i), image_depth)
            cv2.imwrite("temp/{}/model_generation/image_{:1}_normal.png".format(workspace_name, i), image_normal)
            del image_rgba, image_depth, image_normal
        
    clear_memory()
        
    try:
        generate_model(opt)
    except:
        delete_directory("workspaces/{}".format(workspace_name))
    
    os.makedirs("workspaces/{}/images".format(workspace_name), exist_ok=True)
    image = cv2.cvtColor(cv2.imread("temp/{}/six_view_generation/image_0.png".format(workspace_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    cv2.imwrite("workspaces/{}/images/image_0.png".format(workspace_name), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    if not single_image:    
        for i in range(1, 6):
            image = cv2.cvtColor(cv2.imread("temp/{}/six_view_generation/image_{:1}.png".format(workspace_name, i), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            cv2.imwrite("workspaces/{}/images/image_{:1}.png".format(workspace_name, i), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    
    os.makedirs("temp/{}/workspace_manager".format(workspace_name), exist_ok=True)
    video = cv2.VideoCapture("workspaces/{}/results/df_ep{:04d}_rgb.mp4".format(workspace_name, max_epoch))
    image = video.read()[1]
    cv2.imwrite("temp/{}/workspace_manager/df_ep{:04d}_{:04d}_rgb.png".format(workspace_name, max_epoch, 0), image)
    for i in range(1, dataset_size_test):
        temp = video.read()[1]
        cv2.imwrite("temp/{}/workspace_manager/df_ep{:04d}_{:04d}_rgb.png".format(workspace_name, max_epoch, i), temp)
        
    with zipfile.ZipFile("temp/{}/{}.zip".format(workspace_name, workspace_name), "w") as file:
        file.write("workspaces/{}/mesh/albedo.png".format(workspace_name), arcname="albedo.png")
        file.write("workspaces/{}/mesh/mesh.mtl".format(workspace_name), arcname="mesh.mtl")
        file.write("workspaces/{}/mesh/mesh.obj".format(workspace_name), arcname="mesh.obj")
        
    data = {}
    with open("workspaces/{}/info.json".format(workspace_name), "w") as file:
        data["workspace_name"] = workspace_name
        data["dmtet"] = False
        data["single_image"] = single_image
        data["seed"] = int(seed)
        data["max_epoch"] = int(max_epoch)
        data["backbone"] = settings["backbone"]
        data["optim"] = settings["optimizer"]
        data["fp16"] = settings["fp16"]
        data["size"] = int(size)
        data["iters"] = int(iters)
        data["lr"] = float(lr)
        data["batch_size"] = int(batch_size)
        data["dataset_size_train"] = int(dataset_size_train)
        data["dataset_size_valid"] = int(dataset_size_valid)
        data["dataset_size_test"] = int(dataset_size_test)
        json.dump(data, file, indent=4)
    
    clear_memory()
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), gr.Slider(minimum=0, maximum=dataset_size_train-1, label="slide to view generated model from different angles", step=1, interactive=True), gr.File(value="temp/{}/{}.zip".format(workspace_name, workspace_name), visible=True)

def model_finetuning_handler(random_seed, seed, size, tet_grid_size, iters, lr, batch_size, dataset_size_train, dataset_size_valid, dataset_size_test):
    global settings
    global current_tab
    global workspace_name
    global zero123_weights
    
    current_tab = 6
    
    # load opt from opt.json to parse into the trainer object in util.py
    opt = {}
    with open("opt.json", "r") as f:
        opt = json.load(f)
        
    # these arguments are for zero123
    opt["backbone"] = settings["backbone"]
    opt["optim"] = settings["optimizer"]
    opt["fp16"] = settings["fp16"]
    opt["workspace"] = "workspaces/{}_dmtet".format(workspace_name)
    if random_seed:
        opt["seed"] = None
    else:
        opt["seed"] = seed
    opt["h"] = size
    opt["w"] = size
    opt["iters"] = iters
    opt["lr"] = lr
    opt["batch_size"] = batch_size
    opt["dataset_size_train"] = dataset_size_train
    opt["dataset_size_valid"] = dataset_size_valid
    opt["dataset_size_test"] = dataset_size_test
    opt["exp_start_iter"] = opt["exp_start_iter"] or 0
    opt["exp_end_iter"] = opt["exp_end_iter"] or opt["iters"]
    opt["dmtet"] = True
    opt["init_with"] = "workspaces/{}/checkpoints/df.pth".format(workspace_name)
    opt["tet_grid_size"] = int(tet_grid_size)
    opt.pop("full_radius_range")
    opt.pop("full_theta_range")
    opt.pop("full_phi_range")
    opt.pop("full_fovy_range")
    
    if single_image:
        opt["images"] = ["temp/{}_dmtet/model_generation/image_0_rgba.png".format(workspace_name)]
        opt["ref_polars"] = [90.0]
        opt["ref_azimuths"] = [0.0]
        opt["ref_radii"] = [3.2]
        opt["zero123_ws"] = [1]
    else: 
        opt["images"] = ["temp/{}_dmtet/model_generation/image_0_rgba.png".format(workspace_name), "temp/{}_dmtet/model_generation/image_1_rgba.png".format(workspace_name), "temp/{}_dmtet/model_generation/image_2_rgba.png".format(workspace_name), "temp/{}_dmtet/model_generation/image_3_rgba.png".format(workspace_name), "temp/{}_dmtet/model_generation/image_4_rgba.png".format(workspace_name), "temp/{}_dmtet/model_generation/image_5_rgba.png".format(workspace_name)]
        opt["ref_polars"] = [90.0, 90.0, 90.0, 90.0, 180.0, 0.0001]
        opt["ref_azimuths"] = [0.0, 90.0, 180.0, -90.0, 0.0, 0.0]
        opt["ref_radii"] = [3.2, 3.2, 3.2, 3.2, 3.2, 3.2]
        opt["zero123_ws"] = zero123_weights
    
    opt = argparse.Namespace(**opt)
    
    # preprocess and save images to temporary folder in order to make calls to stable-dreamfusion trainer without making changes to its code
    os.makedirs("temp/{}_dmtet/model_generation".format(workspace_name), exist_ok=True)
    image = cv2.cvtColor(cv2.imread("temp/{}/six_view_generation/image_0.png".format(workspace_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    image_rgba, image_depth, image_normal = preprocess(image, size=1024)
    cv2.imwrite("temp/{}_dmtet/model_generation/image_0_rgba.png".format(workspace_name), cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite("temp/{}_dmtet/model_generation/image_0_depth.png".format(workspace_name), image_depth)
    cv2.imwrite("temp/{}_dmtet/model_generation/image_0_normal.png".format(workspace_name), image_normal)
    del image_rgba, image_depth, image_normal
    if not single_image:
        for i in range(1, 6):
            image = cv2.cvtColor(cv2.imread("temp/{}/six_view_generation/image_{:1}.png".format(workspace_name, i), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            image_rgba, image_depth, image_normal = preprocess(image, size=1024)
            cv2.imwrite("temp/{}_dmtet/model_generation/image_{:1}_rgba.png".format(workspace_name, i), cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA))
            cv2.imwrite("temp/{}_dmtet/model_generation/image_{:1}_depth.png".format(workspace_name, i), image_depth)
            cv2.imwrite("temp/{}_dmtet/model_generation/image_{:1}_normal.png".format(workspace_name, i), image_normal)
            del image_rgba, image_depth, image_normal
    
    clear_memory()
    
    try:
        generate_model(opt)
    except:
        delete_directory("workspaces/{}_dmtet".format(workspace_name))
    
    os.makedirs("workspaces/{}_dmtet/images".format(workspace_name), exist_ok=True)
    image = cv2.cvtColor(cv2.imread("temp/{}/six_view_generation/image_0.png".format(workspace_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    cv2.imwrite("workspaces/{}_dmtet/images/image_0.png".format(workspace_name), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    if not single_image:    
        for i in range(1, 6):
            image = cv2.cvtColor(cv2.imread("temp/{}/six_view_generation/image_{:1}.png".format(workspace_name, i), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            cv2.imwrite("workspaces/{}_dmtet/images/image_{:1}.png".format(workspace_name, i), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    
    os.makedirs("temp/{}_dmtet/workspace_manager".format(workspace_name), exist_ok=True)
    video = cv2.VideoCapture("workspaces/{}_dmtet/results/df_ep{:04d}_rgb.mp4".format(workspace_name, max_epoch))
    image = video.read()[1]
    cv2.imwrite("temp/{}_dmtet/workspace_manager/df_ep{:04d}_{:04d}_rgb.png".format(workspace_name, max_epoch, 0), image)
    for i in range(1, dataset_size_test):
        temp = video.read()[1]
        cv2.imwrite("temp/{}_dmtet/workspace_manager/df_ep{:04d}_{:04d}_rgb.png".format(workspace_name, max_epoch, i), temp)
        
    with zipfile.ZipFile("temp/{}_dmtet/{}_dmtet.zip".format(workspace_name, workspace_name), "w") as file:
        file.write("workspaces/{}_dmtet/mesh/albedo.png".format(workspace_name), arcname="albedo.png")
        file.write("workspaces/{}_dmtet/mesh/mesh.mtl".format(workspace_name), arcname="mesh.mtl")
        file.write("workspaces/{}_dmtet/mesh/mesh.obj".format(workspace_name), arcname="mesh.obj")
        
    data = {}
    with open("workspaces/{}_dmtet/info.json".format(workspace_name), "w") as file:
        data["workspace_name"] = "{}_dmtet".format(workspace_name)
        data["dmtet"] = True
        data["single_image"] = single_image
        data["seed"] = int(seed)
        data["max_epoch"] = int(max_epoch)
        data["backbone"] = settings["backbone"]
        data["optim"] = settings["optimizer"]
        data["fp16"] = settings["fp16"]
        data["size"] = int(size)
        data["iters"] = int(iters)
        data["lr"] = float(lr)
        data["batch_size"] = int(batch_size)
        data["dataset_size_train"] = int(dataset_size_train)
        data["dataset_size_valid"] = int(dataset_size_valid)
        data["dataset_size_test"] = int(dataset_size_test)
        json.dump(data, file, indent=4)

    clear_memory()
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), gr.Slider(minimum=0, maximum=dataset_size_train-1, label="slide to view generated model from different angles", step=1, interactive=True), gr.File(value="temp/{}_dmtet/{}_dmtet.zip".format(workspace_name, workspace_name), visible=True)

if __name__ == "__main__":
    main()