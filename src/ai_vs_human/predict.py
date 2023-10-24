from tracker import CenterPointTracker
from utils import read_yaml
from ultralytics import YOLO
import os
import cv2
import numpy as np
import gdown as gd
from tqdm import tqdm



class Track:

    def __init__(self,):
        self.config_content = read_yaml()
        self.cpt_1 = CenterPointTracker()

    def writer(self,input_file_path:str, output_file_path:str):

        self.cpt = CenterPointTracker()
        self.first_frame =  True
        self.ball_id = None
        self.c_30 = True
        cap = cv2.VideoCapture(input_file_path)
        print(f"out file path {output_file_path}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_file_path,fourcc,self.fps,
                         (self.w,
                         self.h))
        self.first_frame=True
        for c in tqdm(range(self.fc)):
            suc,frame = cap.read()
            frame=self.process(frame,c)
            writer.write(frame)
        cap.release()
        writer.release()
    def w_h_fps_fc_initializer(self, input_video_path: str):
        """
        Initialize video-related parameters (width, height, frame count, fps).
        """
        cap = cv2.VideoCapture(input_video_path)
        self.w, self.h, self.fc, self.fps = np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                      cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                                      cap.get(cv2.CAP_PROP_FRAME_COUNT),
                                                      cap.get(cv2.CAP_PROP_FPS)], dtype=np.int32)
        cap.release()

    def download_model(self,model_drive_ids: list[str],out_file_paths:list[str]):
        for model_drive_id,out_file_path in zip(model_drive_ids,out_file_paths):
            if not os.path.exists(out_file_path):
                print(out_file_path)
                gd.download(id = model_drive_id , output = out_file_path )
            else:
                print(f"MODEL ALREADY EXISTS {out_file_path} ")

    def load_model(self,glass_model_path:str,ball_model_path):
        if not all([ os.path.exists(p) for p in [glass_model_path,ball_model_path]]):
            raise FileNotFoundError("Model not found")
        self.glass_model = YOLO(glass_model_path)
        self.ball_model = YOLO(ball_model_path)

    def ball_cup_id(self,ids,bboxes,ball_bbox):
        final_list = []
        for id,bbox in zip(ids,bboxes):
            # print(bbox,ball_bbox)
            (p1_x1,p1_y1,p1_x2,p1_y2),(p2_x1,p2_y1,p2_x2,p2_y2) = bbox,ball_bbox
            p1,p2 = [(p1_x1+p1_x2)/2,(p1_y1+p1_y2)/2],[(p2_x1+p2_x2)/2,(p2_y1+p2_y2)/2]
            dist = self.cpt_1.eul_dist(np.array(p1),np.array(p2))
            final_list.append(  (dist,id,bbox)  )
        return min(final_list)

    def right_or_wrong(self,predicted_id,ids,bboxes,ball_bbox):
        correct_id = self.ball_cup_id(ids,bboxes,ball_bbox)[1]
        return correct_id==predicted_id

    def process(self,frame:np.ndarray,c)->np.ndarray:

        # first_frame =  True
        # ball_id = None
        # c_30 = True
            # for p in all_points:
            #   p1,p2 = p
            #   cv2.rectangle(frame,p1,p2,(0,0,0),3)
            result = self.glass_model.predict(frame,verbose = False)
            ball_result = self.ball_model.predict(frame,verbose = False)
            c_points = []
            bboxes = []
            data = result[0].boxes.data
            for d in data:
                x1,y1,x2,y2,conf,cls = d
                x1,y1,x2,y2,cls = [ int(item.item()) for item in (x1,y1,x2,y2,cls) ]
                if not cls:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),1)
                    c_points.append( np.array( [(x1+x2)/2,(y1+y2)/2] ) )
                    bboxes.append((x1,y1,x2,y2))
                else:print(cls)
            if len(c_points)==3:
                all_ids,all_bboxes=self.cpt.tracker(c_points,bboxes,first_frame=self.first_frame)
                # all_ids,all_bboxes=cpt.track(c_points,bboxes)
                self.first_frame=False
                for id,bb in zip(all_ids,all_bboxes):
                    org_p = bb[:2]
                    # org_p = bb["bbox"][:2]
                    cv2.putText( frame,f"# {id}",(org_p[0],org_p[1]-20),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),2  )

            if len(ball_result[0].boxes.data)==1 and  len(c_points)==3:
                # print(ball_result[0].boxes.data)
                d = ball_result[0].boxes.data[0]
                x1,y1,x2,y2,conf,cls = d
                x1,y1,x2,y2,cls = [ int(item.item()) for item in (x1,y1,x2,y2,cls) ]
                # print(all_ids,all_bboxes)
                dist,ball_id_,bbox = self.ball_cup_id(all_ids,all_bboxes,(x1,y1,x2,y2))
                if self.ball_id is None:self.ball_id = ball_id_ 
                if c >= self.fc-30:
                
                    if self.c_30:
                        print(f" last prediction id {ball_id_}")
                        # user_prediction = int(input("ENTER YOUR PREDICTION : "))
                        self.user_prediction = 1
                        self.user_right_or_wrong = self.right_or_wrong(self.user_prediction,all_ids,all_bboxes,(x1,y1,x2,y2))
                        self.ai_right_or_wrong = self.ball_id==ball_id_
                        self.c_30 = False
                    cv2.putText(frame,f"# USER POINT {int(self.user_right_or_wrong)}",(40,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
                    cv2.putText(frame,f"# AI POINT {int(self.ai_right_or_wrong)}",(40,70),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
                    u_x1,u_y1,u_x2,u_y2 = all_bboxes[all_ids.index(self.user_prediction)]
                    ai_x1,ai_y1,ai_x2,ai_y2 = all_bboxes[all_ids.index(self.ball_id)]
                    cv2.rectangle( frame , (u_x1-15,u_y1-10), (u_x2+15,u_y2+10) , (255,0,0),2 )
                    cv2.rectangle( frame , (ai_x1-15,ai_y1-10), (ai_x2+15,ai_y2+10) , (0,0,255),2 )

            return frame

    
    def combine_all(self):

        artifact_con = self.config_content.get("artifact")
        artifact_root_dir =  artifact_con.get("root_dir")

        ai_vs_human_con = self.config_content.get("ai_vs_human")
        ai_vs_human_root_dir = os.path.join(
            artifact_root_dir,
            ai_vs_human_con.get("root_dir"))

        os.makedirs(ai_vs_human_root_dir,exist_ok=True)

        input_file_path =  os.path.join(
            ai_vs_human_root_dir,
            ai_vs_human_con.get("input_video_file_name"))
        output_file_path =  os.path.join(
            ai_vs_human_root_dir,
            ai_vs_human_con.get("output_video_file_name"))
        glass_model_drive_id = ai_vs_human_con.get("glass_model_drive_id")
        ball_model_drive_id = ai_vs_human_con.get("ball_model_drive_id")
        glass_model_file_path =  os.path.join(
            ai_vs_human_root_dir,
            ai_vs_human_con.get("glass_model_file_name"))
        ball_model_file_path =  os.path.join(
            ai_vs_human_root_dir,

            ai_vs_human_con.get("ball_model_file_name"))

        self.download_model(model_drive_ids = [glass_model_drive_id,ball_model_drive_id], 
                            out_file_paths=[glass_model_file_path,ball_model_file_path])
        
        self.load_model(glass_model_file_path, ball_model_file_path)

        self.w_h_fps_fc_initializer(input_file_path)

        self.writer(input_file_path,output_file_path)




if __name__ == "__main__":
    track = Track()
    track.combine_all()

        
        
        
