import os
import json
import numpy as np
import cv2 as cv

root = os.path.join("data", "nerf", "hand_v2_no_bg")
output = os.path.join('.', root, 'calib', 'transforms.json')
f = open(os.path.join('.', root, 'calib', 'transforms_optim.json'))

jd = json.load(f)

for i in range(len(jd['frames'])):
    print(jd['frames'][i]['file_path'])
    fx = jd['frames'][i]['fl_x']
    fy = jd['frames'][i]['fl_y']
    cx = jd['frames'][i]['cx']
    cy = jd['frames'][i]['cy']
    pose = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    k1 = jd['frames'][i]['k1']
    k2 = jd['frames'][i]['k2']
    p1 = jd['frames'][i]['p1']
    p2 = jd['frames'][i]['p2']
    dist = np.array([k1, k2, p1, p2])
	# update pose
    new_pose = cv.getOptimalNewCameraMatrix(pose, dist, [1280, 720], alpha=1)
    for j in range(0, 2, 1):
        path = jd['frames'][i]['file_path']
        cam, time = path.split('/')[-1].split('_')
        time = int(time.split('.')[0])+j
        print(root, cam, time)
        src = cv.imread(os.path.join(root, "image", "segmented", f"{cam}_{time:04d}.png"), cv.IMREAD_UNCHANGED)
        dst = cv.undistort(src, pose, dist, None, new_pose)
        cv.imwrite(os.path.join(".", root, "image", "undist", f"{cam}_{time:04d}.png"), dst)
        print(os.path.join(".", root, "image", "undist", f"{cam}_{time:04d}.png"))
    jd['frames'][i]['fl_x'] = new_pose[0, 0]
    jd['frames'][i]['fl_y'] = new_pose[1, 1]
    jd['frames'][i]['cx'] = new_pose[0, 2]
    jd['frames'][i]['cy'] = new_pose[1, 2]
    # remove k1k2p1p2
    del jd['frames'][i]['k1']
    del jd['frames'][i]['k2']
    del jd['frames'][i]['p1']
    del jd['frames'][i]['p2']
    # update path
    jd['frames'][i]['file_path'] = os.path.join('image', 'undist', path.split('/')[-1])

print(f"writing {output}")
with open(output, "w") as outfile:
    json.dump(jd, outfile, indent=2)
