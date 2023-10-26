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
        self.all_rect_points = [(77,78,236,292),(240,79,399,293),(405,79,591,292)]
        self.rect_color = {1: (173, 255, 230), 0: (144, 255, 144)}
        # self.rect_colors = {1:(0, 255, 0),0: (0, 0, 255)}
        self.solid_rect_colors = {1:(0, 255, 0),0: (0, 0, 255)}
        self.right_wrong_emoji={1:"Hey Win!",0:"Oops Better Luck Next Time!"}

    def _helper_zone_darw(self,frame):
        for idx,p in enumerate(self.all_rect_points,start=1):
            x1,y1,x2,y2 = p
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,0),1)
            cv2.putText(frame,f"Zone {str(idx)}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)
        return frame

    def input_video_display(self,input_file_path:str,input_display_file_path:str,if_frame = False):
        
        cap = cv2.VideoCapture(input_file_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(input_display_file_path,fourcc,30,
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            suc,frame = cap.read()
            frame = self._helper_zone_darw(frame)
            writer.write(frame)
        writer.release()
        cap.release()

    def writer(self,input_file_path:str, output_file_path:str,user_prediction:int):
        self.win_txt = []
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
            frame=self.process(frame,c,user_prediction)
            writer.write(frame)
        cap.release()
        writer.release()
        return self.win_txt
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

    def zone_cup(self,zone_id,out_result):
        final_list = []
        zone_coor = self.all_rect_points[zone_id-1]
        p2_x1,p2_y1,p2_x2,p2_y2 = zone_coor
        for p in out_result:
            print(p)
            p1_x1,p1_y1,p1_x2,p1_y2,_,_ = p
            p1,p2 = [(p1_x1+p1_x2)/2,(p1_y1+p1_y2)/2],[(p2_x1+p2_x2)/2,(p2_y1+p2_y2)/2]
            dist = self.cpt_1.eul_dist(np.array(p1),np.array(p2))
            final_list.append(  (dist,p)  )
        return min(final_list)



    def right_or_wrong(self,predicted_id,ids,bboxes,ball_bbox):
        correct_id = self.ball_cup_id(ids,bboxes,ball_bbox)[1]
        return correct_id==predicted_id

    def find_user_win_or_not(self,ball_id_bboox=None,all_zone_coor=None,user_prediction=None):
        bx1,by1,bx2,by2 = ball_id_bboox
        p1 = (bx1+by1)/2, (bx2+by2)/2
        final_list = []
        for p in all_zone_coor:
            x1,y1,x2,y2 = p
            p2 = (x1+y1)/2, (x2+y2)/2
            dist = self.cpt_1.eul_dist(p1, p2)
            final_list.append((dist,p))
        return final_list
        


    def process(self,frame:np.ndarray,c,user_prediction,put_track_id = False ,draw_zone = False)->np.ndarray:

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
                    if put_track_id:cv2.putText( frame,f"# {id}",(org_p[0],org_p[1]-20),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0),2  )

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
                        self.user_prediction = user_prediction
                        
                        self.user_bbox = [int(i.item()) for i in self.zone_cup(self.user_prediction, result[0].boxes.data)[-1][:-2]]
                        self.user_right_or_wrong = self.user_bbox == list(bbox)
                        print(f"self.user_right_or_wrong {self.user_right_or_wrong} , {self.user_bbox} , {bbox} ")
                        self.ai_right_or_wrong = self.ball_id==ball_id_
                        self.c_30 = False
                    cv2.putText(frame,f"# USER POINT : {int(self.user_right_or_wrong)}",(20,20),cv2.FONT_HERSHEY_COMPLEX,0.4,self.solid_rect_colors[self.user_right_or_wrong],1)
                    cv2.putText(frame,f"# AI POINT : {int(self.ai_right_or_wrong)}",(20,40),cv2.FONT_HERSHEY_COMPLEX,0.4,self.solid_rect_colors[self.ai_right_or_wrong],1)
                    # u_x1,u_y1,u_x2,u_y2 = all_bboxes[all_ids.index(self.user_prediction)]
                    u_x1,u_y1,u_x2,u_y2 = self.all_rect_points[user_prediction-1]
                    ai_x1,ai_y1,ai_x2,ai_y2 = all_bboxes[all_ids.index(self.ball_id)]
                    frame = self.fill_color( (ai_x1,ai_y1,ai_x2,ai_y2),self.ai_right_or_wrong,frame )
                    frame = self.fill_color( self.user_bbox,self.user_right_or_wrong,frame )

                    for p_,t_o_f,a_u in zip([self.user_bbox,(ai_x1,ai_y1,ai_x2,ai_y2)],[self.ai_right_or_wrong,self.user_right_or_wrong],["AI","USER"]):
                        _x1,_y1,_x2,_y2 = p_
                        c_y1 =  _y1-15 if a_u == "USER" else _y1
                        txt = f"# {a_u} PREDICTION {self.right_wrong_emoji[t_o_f]}"
                        self.win_txt.append(txt)
                        cv2.putText( frame, txt,(_x1,c_y1-15),cv2.FONT_HERSHEY_COMPLEX,0.4,self.solid_rect_colors[t_o_f],1)
                        cv2.rectangle( frame , (_x1,_y1), (_x2,_y2) ,self.solid_rect_colors[t_o_f] ,1 )
                    # cv2.rectangle( frame , (ai_x1-15,ai_y1-10), (ai_x2+15,ai_y2+10) , (0,0,255),2 )
            
            return self._helper_zone_darw(frame)

    

    def fill_color(self,coor, right_or_wrong, frame):
        x1, y1, x2, y2 = coor
        sub_img = frame[y1:y2, x1:x2]

        # Create an image with the same shape as sub_img filled with the specified color
        rc = np.zeros_like(sub_img)
        rc[:, :] = self.rect_color.get(right_or_wrong, (255, 255, 255))

        res = cv2.addWeighted(sub_img, 0.6, rc, 0.5, 1.0)

        # Putting the image back to its position
        frame[y1:y2, x1:x2] = res

        return frame
    def combine_all(self,user_prediction:int):

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

        win_list = self.writer(input_file_path,output_file_path,user_prediction)
        return win_list


if __name__ == "__main__":
    track = Track()
    track.combine_all(3)
        
        
        
