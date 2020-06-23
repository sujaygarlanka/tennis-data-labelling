import sys
import argparse
import cv2
import os


def extractImages(pathIn, pathOut,fps, beg, end):

    vidcap = cv2.VideoCapture(pathIn)

    fps_actual = vidcap.get(cv2.CAP_PROP_FPS)
    print(fps_actual)

    if fps > fps_actual: 
        fps = fps_actual

    frameRate = (1.0 / fps) #//it will capture image in each 0.5 second
    #print (frameRate)
    count=1
    #success = getFrame(beg)
    hasFrames = True
    while hasFrames and beg <= end:
        #print(beg, ",", end)
        count = count + 1
        beg = beg + frameRate
        beg = round(beg, 2)

        vidcap.set(cv2.CAP_PROP_POS_MSEC,beg*1000)
        hasFrames,image = vidcap.read()

        tail = os.path.split(pathIn)[1]
        if hasFrames:
            cv2.imwrite(f"{pathOut}/{tail}_{beg}_{end}_{frameRate}.jpg", image)     # save frame as JPG file

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--vid", help="path to video")
    a.add_argument("--out", help="path to images")
    a.add_argument("--fps", type=int, help="frame rate")
    a.add_argument("--beg", type=int, help="beginning")
    a.add_argument("--end", type=int, help="ending")
    args = a.parse_args()
    print(args)
    extractImages(args.vid, args.out,args.fps,args.beg,args.end)

