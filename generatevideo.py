import cv2
import numpy as np
import os
import glob
import ffmpeg
import argparse
import shutil
from PIL import Image, ImageEnhance
from pydub import AudioSegment

temp_paths = ["temp", "./temp/blured", "./temp/frames"]
input_paths = []

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-images_folder")
    parser.add_argument("-audio")
    parser.add_argument("-output_video")
    args = parser.parse_args()    
    if(args.images_folder is not None and args.audio is not None and args.output_video is not None):
        images_folder_path = args.images_folder
        audio_path = args.audio
        output_video_path = args.output_video
        is_images = os.path.isdir(images_folder_path)
        is_audio = os.path.isfile(audio_path)
        if(is_images):
            print("Images folder exists.")
            if(is_audio):
                print("Audio file exists.")
                input_paths.append(images_folder_path)
                input_paths.append(audio_path)
                input_paths.append(output_video_path)
            else:
                print("Please set correct audio file path.")
        else:
            print("Please set correct image folder path.")  


def createtemproraryPaths():
    for path in temp_paths:
        os.mkdir(path)

def convert_png_to_jpg(arg):
    png_images = glob.glob("{}/*.png".format(arg))
    for j in range(len(png_images)):
        splited_array = png_images[j].split("\\")
        image_name = ''
        splited_element = splited_array[len(splited_array)-1]
        for i in range(0, len(splited_array[len(splited_array)-1])):
            if(splited_element[i] == '.'):
                break
            else:
                image_name = image_name + splited_element[i]
        jpg_image = Image.open(png_images[j])
        jpg_image = jpg_image.convert("RGB")
        jpg_image.save("{}/{}.jpg".format(arg,image_name))
        os.remove("{}/{}.png".format(arg,image_name))

def blur_and_merge(arg):
    
    original_images = glob.glob("{}/*.jpg".format(arg))

    print(original_images)
    scale_percent = 60

    for i in range(0,len(original_images)):
        original_img = cv2.imread(original_images[i])
        blur_img = cv2.GaussianBlur(original_img, (51,51), 0)
        width = 710
        height = 720
        dimensions = (width, height)
        resized_small = cv2.resize(original_img, dimensions, interpolation = cv2.INTER_AREA)
        width = 1280
        height = 720
        dimensions = (width, height)
        resized_large = cv2.resize(blur_img, dimensions, interpolation = cv2.INTER_AREA)

        cv2.imwrite('./temp/img1.jpg', resized_small)
        cv2.imwrite('./temp/img2.jpg', resized_large)

        img = Image.open('./temp/img1.jpg')
        img_w, img_h = img.size
        background = Image.open('./temp/img2.jpg')
        bg_w, bg_h = background.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)  
        background.paste(img, offset)
        background.save('./temp/blured/{}.jpg'.format(i))

def change_brightness(img, value):
    im_pil = Image.fromarray(img)    
    enhancer = ImageEnhance.Brightness(im_pil)
    enhanced_im = enhancer.enhance(value)
    im_np = np.asarray(enhanced_im)
    return im_np

def createFrameImages():
    blured_images = glob.glob("temp/blured/*.jpg")
    x = y = 0
    for i in range(0,len(blured_images)):
        load_image = cv2.imread(blured_images[i])
        if i % 2 == 0:
            start_h=483  #720   
            start_w=860  #1280
            os.mkdir("./temp/frames/{}".format(i))
            brightness = 0.959
            brightness2 = 0
            for j in range(0,288):
                crop_img = load_image[int(y):int(y+start_h), int(x):int(x+start_w)]
                start_h = start_h + 0.823
                start_w = start_w + 1.458
                crop_img = cv2.resize(crop_img, (1280,720), interpolation = cv2.INTER_AREA)
                if(j < 24):
                    newcropimg = change_brightness(crop_img, brightness2)
                    cv2.imwrite('./temp/frames/{}/{}.jpg'.format(i,j), newcropimg)
                    brightness2 = brightness2 + 0.041
                elif(j > 263):
                    newcropimg = change_brightness(crop_img, brightness)
                    cv2.imwrite('./temp/frames/{}/{}.jpg'.format(i,j), newcropimg)
                    brightness = brightness - 0.041
                else:
                    cv2.imwrite('./temp/frames/{}/{}.jpg'.format(i,j), crop_img)
        else:
            start_h=720 
            start_w=1280
            os.mkdir("./temp/frames/{}".format(i))
            brightness = 0
            brightness2 = 0.959
            for k in range(0,288):
                crop_img = load_image[int(y):int(y+start_h), int(x):int(x+start_w)]
                start_h = start_h - 0.823
                start_w = start_w - 1.458
                crop_img = cv2.resize(crop_img, (1280,720), interpolation = cv2.INTER_AREA)
                if(k < 24):
                    newcropimg = change_brightness(crop_img, brightness)
                    cv2.imwrite('./temp/frames/{}/{}.jpg'.format(i,k), newcropimg)
                    brightness = brightness + 0.041
                elif(k > 263):
                    newcropimg = change_brightness(crop_img, brightness2)
                    cv2.imwrite('./temp/frames/{}/{}.jpg'.format(i,k), newcropimg)
                    brightness2 = brightness2 - 0.041
                else:
                    cv2.imwrite('./temp/frames/{}/{}.jpg'.format(i,k), crop_img)

def createVideo():
    frame_folders = glob.glob("./temp/frames/*")
    video_name = "temp/video.mp4".format()
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), 24, (1280,720))

    for i in range(0, len(frame_folders)):
        frames = glob.glob("./temp/frames/{}/*jpg".format(i))
        for j in range(0, len(frames)):
            writeable_image = cv2.imread('./temp/frames/{}/{}.jpg'.format(i,j))
            video.write(writeable_image)
        
    cv2.destroyAllWindows()
    video.release()

def add_audio_to_video(audio,video):
    input_video = ffmpeg.input("temp/video.mp4")
    video_duration = int(float(ffmpeg.probe('temp/video.mp4')['format']['duration']))
    audio_duration = int(float(ffmpeg.probe('{}'.format(audio))['format']['duration']))
    end_time = video_duration * 1000
    if (audio_duration > video_duration):
        song = AudioSegment.from_mp3("{}".format(audio))
        
        audio = song[:end_time]
        audio.export("./temp/temp.mp3", format="mp3")
    else:
        diff = video_duration - audio_duration
        if(diff > audio_duration):
            count_of_diff = int(diff / audio_duration + 1)
            song = AudioSegment.from_mp3("{}".format(audio))
            new_song = song
            for i in range(0, count_of_diff):
                new_song += song
            audio = new_song[:end_time]
            audio.export("./temp/temp.mp3", format="mp3")
    added_audio = ffmpeg.input("./temp/temp.mp3")
    (
        ffmpeg
        .concat(input_video, added_audio, v=1, a=1)
        .output("{}".format(video))
        .run(overwrite_output=True)
    )

if(__name__ == "__main__"):
    get_arguments()
    print("Please wait, generating your video....")
    createtemproraryPaths()
    convert_png_to_jpg(input_paths[0])
    blur_and_merge(input_paths[0])
    createFrameImages()
    createVideo()
    add_audio_to_video(input_paths[1], input_paths[2])
    shutil.rmtree("temp", ignore_errors=False, onerror=None)
    print("Your video is created.")


    



