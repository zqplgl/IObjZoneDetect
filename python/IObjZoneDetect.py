import _IObjZoneDetect as od
import os
import cv2

class IObjZoneYOLOV3Detect:
    def __init__(self, cfg_file, weight_file, gpu_id=0):
        self.__detector = od.ObjZoneDetector(cfg_file,weight_file,0,0)

    def detect(self,im,confidence_thread = 0.7):
        r = []
        boxes = self.__detector.detect(im,im.shape[1],im.shape[0],confidence_thread)
        for b in boxes:
            temp = {}
            temp["zone"] = (b.zone.x,b.zone.y,b.zone.x+b.zone.w,b.zone.y+b.zone.h)
            temp["cls"] = b.cls
            temp["score"] = b.score
            r.append(temp)
        return r

class ICartwheelZoneDetect:
    def __init__(self,model_dir, gpu_id=0):
        if not model_dir.endswith("/"):
            model_dir += "/"
        cfg_file = model_dir + "cartwheel/yolov3.cfg"
        weight_file = model_dir + "cartwheel/yolov3.weights"
        self.__detector = IObjZoneYOLOV3Detect(cfg_file,weight_file,gpu_id)

    def detect(self, im, confidence_threshold=0.7):
        boxes = self.__detector.detect(im,confidence_threshold)

        result = 0
        for box in boxes:
            result += (box["cls"]+1)

        return result,boxes

def addRectangle(im,boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in boxes:
        zone = box["zone"]
        cv2.rectangle(im,(zone[0],zone[1]),(zone[2],zone[3]),(0,0,255),2)
        cv2.putText(im, "%s:%.2f"%(box["cls"],box["score"]),(zone[0],zone[1]),font,1,(0,255,0))

def run():
    cfg_file = "/home/zqp/models/cartwheel/yolo_v3vehicle.cfg"
    weight_file = "/home/zqp/models/cartwheel/yolo_v3vehicle_15000.weights"
    model_dir = "/home/zqp/models/"
    detector = ICartwheelZoneDetect(model_dir)

    picdir = "/home/zqp/testpic/cartwheel/"
    for picname in os.listdir(picdir):
        im = cv2.imread(picdir+picname)
        result,boxes = detector.detect(im)
        addRectangle(im, boxes)
        cv2.imshow("im", im)
        if cv2.waitKey(0)==27:
            break

if __name__=="__main__":
    run()


