"""
This project is developed by Haofan Wang to support face swap in single frame. Multi-frame will be supported soon!

It is highly built on the top of insightface, sd-webui-roop and CodeFormer.
"""
import sys
sys.path.append('./inswapper')

import os
import cv2
import copy
import torch
import argparse
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple


def getFaceSwapModel(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def getFaceAnalyser(model_path: str, providers,
                    det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", root="./checkpoints", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser,
                 frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

    
def get_many_faces(face_analyser,
                   frame:np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper,
              source_face,
              target_face,
              temp_frame):
    """
    paste source_face on target image
    """
    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


def process(source_img: Image.Image,
            target_img: Image.Image,
            model: str):
    # load machine default available providers
    providers = onnxruntime.get_available_providers()

    # load face_analyser
    face_analyser = getFaceAnalyser(model, providers)
    
    # load face_swapper
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)
    
    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    
    # detect single face in the target image
    target_face = get_one_face(face_analyser, target_img)
    
    if target_face:
        temp_frame = copy.deepcopy(target_img)
        
        # detect single face in the source image
        source_face = get_one_face(face_analyser, cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR))

        if source_face:
            print("Swapping single face in target image with single face from source image")
            temp_frame = swap_face(
                face_swapper,
                source_face,
                target_face,
                temp_frame
            )
        else:
            raise Exception("No source face found!")
        
        result = temp_frame
    else:
        print("No target face found!")
        result = target_img  # Default to original target image if no faces are found
    
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image

'''
def parse_args():
    parser = argparse.ArgumentParser(description="Face swap.")
    parser.add_argument("--source_img", type=str, required=True, help="The path of source image.")
    parser.add_argument("--target_img", type=str, required=True, help="The path of target image.")
    parser.add_argument("--output_img", type=str, required=False, default="result.png", help="The path and filename of output image.")
    parser.add_argument("--face_restore", action="store_true", help="The flag for face restoration.")
    parser.add_argument("--background_enhance", action="store_true", help="The flag for background enhancement.")
    parser.add_argument("--face_upsample", action="store_true", help="The flag for face upsample.")
    parser.add_argument("--upscale", type=int, default=1, help="The upscale value, up to 4.")
    parser.add_argument("--codeformer_fidelity", type=float, default=0.5, help="The codeformer fidelity.")
    args = parser.parse_args()
    return args
'''

def swapper_fn(model, source_img, target_img, face_restore, background_enhance, face_upsample, upscale, codeformer_fidelity, output_img):

    
    #source_img_path = args.source_img
    #target_img_path = args.target_img
    
    #source_img = Image.open(source_img_path)
    #target_img = Image.open(target_img_path)

    # download from https://huggingface.co/deepinsight/inswapper/tree/main
    #model = "./checkpoints/inswapper_128.onnx"
    result_image = process(source_img, target_img, model)
    
    if face_restore:

        import restoration 
        
        # make sure the ckpts downloaded successfully
        #check_ckpts()
        
        # https://huggingface.co/spaces/sczhou/CodeFormer
        upsampler = restoration.set_realesrgan()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        codeformer_net = restoration.ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                         codebook_size=1024,
                                                         n_head=8,
                                                         n_layers=9,
                                                         connect_list=["32", "64", "128", "256"],
                                                        ).to(device)
        ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()
        
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = restoration.face_restoration(result_image, 
                                        background_enhance, 
                                        face_upsample, 
                                        upscale, 
                                        codeformer_fidelity,
                                        upsampler,
                                        codeformer_net,
                                        device)
        result_image = Image.fromarray(result_image)
    
    # save result
    result_image.save(output_img)
    print(f'Result saved successfully: {output_img}')
    return result_image
