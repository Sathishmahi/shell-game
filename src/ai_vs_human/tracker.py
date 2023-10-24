import numpy as np
from scipy.optimize import linear_sum_assignment


class CenterPointTracker:
  def __init__(self):
    self.final_dict = { i:{"point":None,"bbox":None} for i in range(1,4) }

  @staticmethod
  def eul_dist(p1,p2):
    return np.sqrt(np.sum(np.square(p2-p1)))
  def tracker(self,points,bboxs,first_frame =  True):
    if first_frame:
      all_id,all_bb = [],[]
      for c,(p,bb) in enumerate(zip(points,bboxs),start=1):
        self.final_dict[c]["point"] = p  
        self.final_dict[c]["bbox"] = bb
        all_id.append(c)
        all_bb.append(bb)
      return all_id,all_bb
      
    else:
      all_id,all_bb = [],[] 
      for p,bb in zip(points,bboxs):
        all_dist = []
        for key,val in self.final_dict.items():
          p1,p2 = p,val["point"]
          dist = self.eul_dist(p1,p2)
          all_dist.append((dist,key,p,bb))
        dist,k,new_p,new_bb = min(all_dist)
        all_id.append(k)
        all_bb.append(bb)
        self.final_dict[k]["point"]=new_p
        self.final_dict[k]["bbox"]=new_bb
      return all_id,all_bb



class CenterPointTracker1:

  def __init__(self):
    self.next_id = 1
    self.objects = {}

  def track(self, points, bboxes):
    new_objects = {}
    
    if len(self.objects) == 0:
      # First frame, assign new IDs
      for p, bb in zip(points, bboxes):
        new_objects[self.next_id] = {'point': p, 'bbox': bb}
        self.next_id += 1

    else:
      # Calculate cost matrix between old and new points
      cost_matrix = np.zeros((len(self.objects), len(points)))
      for i, obj_id in enumerate(self.objects):
        for j, new_p in enumerate(points):
          obj = self.objects[obj_id]
          cost_matrix[i,j] = np.linalg.norm(obj['point'] - new_p)
          
      # Solve assignment problem  
      row_ind, col_ind = linear_sum_assignment(cost_matrix)
      
      # Update tracking IDs
      for r, c in zip(row_ind, col_ind):
        obj_id = list(self.objects.keys())[r]
        new_objects[obj_id] = {'point': points[c], 'bbox': bboxes[c]}
        
    # Update objects
    self.objects = new_objects

    return list(new_objects.keys()), list(new_objects.values())