
import cv2
import os
import numpy as np
from PIL import Image
import paddlehub as hub


def split_video_to_frames(video_file_path, frames_folder_path):
    print("call split_video_to_frames")

    if not os.path.exists(video_file_path):
        print(f"video file {video_file_path} do not exist.")
        return

    cap = cv2.VideoCapture(video_file_path)
    index = 0
    while True:
        ret, frame = cap.read()
        print(f"capture ret={ret} frame={frame}")
        if ret:
            cv2.imwrite(f'{frames_folder_path}/{index}.jpg', frame)
            print(type(frame))
            print(frame.shape)
            index += 1
        else:
            break

    cap.release()
    print('video split finish, all %d frame' % index)


def turn_frames_to_humans(frames_folder_path, humans_folder_path):
    print("call turn_frames_to_humans")

    print(f"frames_folder_path = {frames_folder_path}")
    print(f"humans_folder_path = {humans_folder_path}")

    print(os.listdir(frames_folder_path))

    # load model
    module = hub.Module(name="deeplabv3p_xception65_humanseg")

    test_img_path = [os.path.join(frames_folder_path, fname) for fname in os.listdir(frames_folder_path)]
    input_dict = {"image": test_img_path}

    results = module.segmentation(data=input_dict, output_dir=humans_folder_path)
    for result in results:
        print(result)


def blend_one_human_with_background(one_human_image_path, background_image_path, one_blended_image_path):
    print("call blend_one_human_with_background")

    background_image = Image.open(background_image_path).convert('RGB')

    one_human_image = Image.open(one_human_image_path).resize(background_image.size)

    # PNG format = RGBA
    one_human_image = np.array(one_human_image)
    print(one_human_image.shape)
    print(one_human_image[0, 0])

    # transparency dimension of A in RGBA
    one_human_image_A = one_human_image[:, :, -1]
    # print(one_human_image_A.shape)
    # print(one_human_image_A[0, 0])
    # print(list(set(one_human_image_A.ravel())))

    # RGB dimension in RGBA
    one_human_image_RGB = one_human_image[:, :, :3]

    scope_map = one_human_image_A / 255
    # print(f"scope_map.shape={scope_map.shape}")
    # print(scope_map[0, 0])
    # print(list(set(scope_map.ravel())))

    scope_map = scope_map[:, :, np.newaxis]
    # print(f"scope_map.shape={scope_map.shape}")
    # print(scope_map[0, 0])

    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    # print(f"scope_map.shape={scope_map.shape}")
    # print(scope_map[0, 0])

    human_layer = np.multiply(scope_map, one_human_image_RGB)
    backgroud_layer = np.multiply((1 - scope_map), np.array(background_image))
    blended_image = human_layer + backgroud_layer
    
    blended_image = Image.fromarray(np.uint8(blended_image))
    blended_image.save(one_blended_image_path)


def blend_humans_with_background(humans_folder_path, background_image_path, frames_blended_folder_path):
    print("call blend_humans_with_background")

    all_human_image_paths = [filename for filename in os.listdir(humans_folder_path)]

    for i, one_human_image_name in enumerate(all_human_image_paths):
        one_human_image_path = f"{humans_folder_path}{one_human_image_name}"
        print(f"one_human_image_path = {one_human_image_path}")

        if not os.path.exists(one_human_image_path):
            print(f"one human image({one_human_image_path}) does not exist.")
            continue

        one_blended_image_path = f"{frames_blended_folder_path}{i}.png"
        print(one_blended_image_path)

        blend_one_human_with_background(one_human_image_path, background_image_path, one_blended_image_path)
   

def init_canvas(width, height, color=(255, 255, 255)):
    print("call init_canvas")

    canvas = np.ones((height, width, 3), dtype="uint8")
    # assign all element with specific color
    canvas[:] = color
    return canvas


def make_background_file(width, height, out_path):
    canvas = init_canvas(width, height, color=(0, 255, 0))
    cv2.imwrite(out_path, canvas)


def concatenate_frames_blended(frames_blended_folder_path, video_blended_file_path, size):
    print("call concatenate_frames_blended")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_blended_file_path, fourcc, 3.0, size)
    files = os.listdir(frames_blended_folder_path)

    for i in range(len(files)):
        one_frame_blended = frames_blended_folder_path + '%d.png' % i
        if not os.path.exists(one_frame_blended):
            continue

        img = cv2.imread(one_frame_blended)
        out.write(img)
    out.release()


# Config
video_path = 'workspace/sample.mp4'
video_blended_path = 'workspace/output.mp4'
background_image_path = 'workspace/green.jpg'

frames_folder_path = 'workspace/frames/'
humans_folder_path = 'workspace/humans/'
frames_blended_folder_path = 'workspace/frames_blended/'

background_size = (1920, 1080)

if __name__ == "__main__":
    print("video to frames")
    if not os.path.exists(frames_folder_path):
        os.mkdir(frames_folder_path)
        split_video_to_frames(video_path, frames_folder_path)

    print("frames to humans")
    if not os.path.exists(humans_folder_path):
        os.mkdir(humans_folder_path)
        turn_frames_to_humans(frames_folder_path, humans_folder_path)

    print("make green background")
    if not os.path.exists(background_image_path):
        make_background_file(*background_size, background_image_path)

    print("blend humans with background")
    if not os.path.exists(frames_blended_folder_path):
        os.mkdir(frames_blended_folder_path)
        blend_humans_with_background(humans_folder_path, background_image_path, frames_blended_folder_path)

    print("concatenate frames blended into video")
    if not os.path.exists(video_blended_path):
        concatenate_frames_blended(frames_blended_folder_path, video_blended_path, background_size)



