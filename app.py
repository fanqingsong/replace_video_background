
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
    while(True):
        ret,frame = cap.read()
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
    print(test_img_path)
    input_dict = {"image": test_img_path}

    results = module.segmentation(data=input_dict, output_dir=humans_folder_path)
    for result in results:
        print(result)


def blend_one_human_with_background(foreground_image, background_image, frames_blended_folder_path):
    """
    将抠出的人物图像换背景
    foreground_image: 前景图片，抠出的人物图片
    background_image: 背景图片
    """

    print("call blend_one_human_with_background")

    # 读入图片
    background_image = Image.open(background_image).convert('RGB')
    foreground_image = Image.open(foreground_image).resize(background_image.size)

    # 图片加权合成
    scope_map = np.array(foreground_image)[:, :, -1] / 255
    scope_map = scope_map[:,:,np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(foreground_image)[:, :, :3]) + np.multiply((1 - scope_map), np.array(background_image))
    
    #保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save(frames_blended_folder_path)


def blend_humans_with_background(humans_folder_path, green_background_file_path, frames_blended_folder_path):
    print("call blend_humans_with_background")

    humanseg_png = [filename for filename in os.listdir(humans_folder_path)]
    for i, img in enumerate(humanseg_png):
        img_path = os.path.join(humans_folder_path + '%d.png' % (i))
        print(img_path)
        output_path_img = frames_blended_folder_path + '%d.png' % i
        print(output_path_img)
        blend_one_human_with_background(img_path, green_background_file_path, output_path_img)
   

def init_canvas(width, height, color=(255, 255, 255)):
    print("call init_canvas")

    canvas = np.ones((height, width, 3), dtype="uint8")
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
        img = cv2.imread(frames_blended_folder_path + '%d.png' % i)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        # img = cv2.resize(img, (1280,720))
        out.write(img)#保存帧
    out.release()


# Config
video_file_path = 'workspace/sample.mp4'
background_file_path = 'workspace/green.jpg'
frames_folder_path = 'workspace/frames/'
humans_folder_path = 'workspace/humans/'
frames_blended_folder_path = 'workspace/frames_blended/'
video_blended_file_path = 'workspace/output.mp4'

if __name__ == "__main__":
    # 第一步：视频->图像
    print("video to frames")
    if not os.path.exists(frames_folder_path):
        os.mkdir(frames_folder_path)
        split_video_to_frames(video_file_path, frames_folder_path)

    # 第二步：抠图
    print("frames to humans")
    if not os.path.exists(humans_folder_path):
        os.mkdir(humans_folder_path)
        turn_frames_to_humans(frames_folder_path, humans_folder_path)

    # 第三步：生成绿幕并合成
    print("make green background")
    if not os.path.exists(background_file_path):
        make_background_file(1920, 1080, background_file_path)

    print("blend humans with background")
    if not os.path.exists(frames_blended_folder_path):
        os.mkdir(frames_blended_folder_path)
        blend_humans_with_background(humans_folder_path, background_file_path, frames_blended_folder_path)

    # 第四步：合成视频
    print("concatenate frames blended into video")
    if not os.path.exists(video_blended_file_path):
        concatenate_frames_blended(frames_blended_folder_path, video_blended_file_path, (1920, 1080))
