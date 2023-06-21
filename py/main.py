import os
import time
import cv2
import numpy as np
from pye3d.camera import CameraModel
from pye3d.detector_3d import Detector3D, DetectorMode

import warnings
warnings.filterwarnings("ignore")

# Old RANSAC (4x slower)

# def fit_rotated_ellipse_ransac(
#     data, iter=5, sample_num=10, offset=80  # 80.0, 10, 80
# ):  # before changing these values, please read up on the ransac algorithm
#     # However if you want to change any value just know that higher iterations will make processing frames slower
#     count_max = 0
#     effective_sample = None

#     # TODO This iteration is extremely slow.
#     #
#     # Either we need to keep the iteration number low, or we need to keep a worker pool specifically
#     # for handling this calculation. It's parallelizable, so just throwing something like joblib at
#     # it would be fine.
#     for i in range(iter):
#         sample = np.random.choice(len(data), sample_num, replace=False)

#         xs = data[sample][:, 0].reshape(-1, 1)
#         ys = data[sample][:, 1].reshape(-1, 1)

#         J = np.mat(
#             np.hstack((xs * ys, ys**2, xs, ys,
#                       np.ones_like(xs, dtype=np.float64)))
#         )
#         Y = np.mat(-1 * xs**2)
#         P = (J.T * J).I * J.T * Y

#         # fitter a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
#         a = 1.0
#         b = P[0, 0]
#         c = P[1, 0]
#         d = P[2, 0]
#         e = P[3, 0]
#         f = P[4, 0]
#         ellipse_model = (
#             lambda x, y: a * x**2 + b * x * y + c * y**2 + d * x + e * y + f
#         )

#         # thresh
#         ran_sample = np.array(
#             [[x, y] for (x, y) in data if np.abs(ellipse_model(x, y)) < offset]
#         )

#         if len(ran_sample) > count_max:
#             count_max = len(ran_sample)
#             effective_sample = ran_sample

#     return fit_rotated_ellipse(effective_sample)


# def fit_rotated_ellipse(data):
#     xs = data[:, 0].reshape(-1, 1)
#     ys = data[:, 1].reshape(-1, 1)

#     J = np.mat(np.hstack((xs * ys, ys**2, xs, ys,
#                np.ones_like(xs, dtype=np.float64))))
#     Y = np.mat(-1 * xs**2)
#     P = (J.T * J).I * J.T * Y

#     a = 1.0
#     b = P[0, 0]
#     c = P[1, 0]
#     d = P[2, 0]
#     e = P[3, 0]
#     f = P[4, 0]
#     theta = 0.5 * np.arctan(b / (a - c))

#     cx = (2 * c * d - b * e) / (b**2 - 4 * a * c)
#     cy = (2 * a * e - b * d) / (b**2 - 4 * a * c)

#     cu = a * cx**2 + b * cx * cy + c * cy**2 - f
#     w = np.sqrt(
#         cu
#         / (
#             a * np.cos(theta)**2
#             + b * np.cos(theta) * np.sin(theta)
#             + c * np.sin(theta)**2
#         )
#     )
#     h = np.sqrt(
#         cu
#         / (
#             a * np.sin(theta)**2
#             - b * np.cos(theta) * np.sin(theta)
#             + c * np.cos(theta)**2
#         )
#     )

#     def ellipse_model(x, y): return a * x**2 + b * x * \
#         y + c * y**2 + d * x + e * y + f

#     error_sum = np.sum([ellipse_model(x, y) for x, y in data])

#     return (cx, cy, w, h, theta)

# New RANSAC (from the new-algos branch)

def ellipse_model(data, y, f):
    """
    There is no need to make this process a function, since making the process a function will slow it down a little by calling it.
    The results may be slightly different from the lambda version due to calculation errors derived from float types, but the calculation results are virtually the same.
    a = 1.0,b = P[0],c = P[1],d = P[2],e = P[3],f = P[4]
    :param data:
    :param y: np.c_[d, e, a, c, b]
    :param f: f == P[4, 0]
    :return: this_return == np.array([ellipse_model(x, y) for (x, y) in data ])
    """
    return data.dot(y) + f

# @profile
def fit_rotated_ellipse_ransac(data: np.ndarray, rng: np.random.Generator, iter=100, sample_num=10, offset=80  # 80.0, 10, 80
                               ):  # before changing these values, please read up on the ransac algorithm
    # However if you want to change any value just know that higher iterations will make processing frames slower
    effective_sample = None
    
    # The array contents do not change during the loop, so only one call is needed.
    # They say len is faster than shape.
    # Reference url: https://stackoverflow.com/questions/35547853/what-is-faster-python3s-len-or-numpys-shape
    len_data = len(data)
    
    if len_data < sample_num:
        return None
    
    # Type of calculation result
    ret_dtype = np.float64
    
    # Sorts a random number array of size (iter,len_data). After sorting, returns the index of sample_num random numbers before sorting.
    # If the array size is less than about 100, this is faster than rng.choice.
    rng_sample = rng.random((iter, len_data)).argsort()[:, :sample_num]
    # or
    # I don't see any advantage to doing this.
    # rng_sample = np.asarray(rng.random((iter, len_data)).argsort()[:, :sample_num], dtype=np.int32)
    
    # I don't think it looks beautiful.
    # x,y,x**2,y**2,x*y,1,-1*x**2
    datamod = np.concatenate(
        [data, data ** 2, (data[:, 0] * data[:, 1])[:, np.newaxis], np.ones((len_data, 1), dtype=ret_dtype),
         (-1 * data[:, 0] ** 2)[:, np.newaxis]], axis=1,
        dtype=ret_dtype)
    
    datamod_slim = np.array(datamod[:, :5], dtype=ret_dtype)
    
    datamod_rng = datamod[rng_sample]
    datamod_rng6 = datamod_rng[:, :, 6]
    datamod_rng_swap = datamod_rng[:, :, [4, 3, 0, 1, 5]]
    datamod_rng_swap_trans = datamod_rng_swap.transpose((0, 2, 1))
    
    # These two lines are one of the bottlenecks
    datamod_rng_5x5 = np.matmul(datamod_rng_swap_trans, datamod_rng_swap)
    datamod_rng_p5smp = np.matmul(np.linalg.inv(datamod_rng_5x5), datamod_rng_swap_trans)
    
    datamod_rng_p = np.matmul(datamod_rng_p5smp, datamod_rng6[:, :, np.newaxis]).reshape((-1, 5))
    
    # I don't think it looks beautiful.
    ellipse_y_arr = np.asarray(
        [datamod_rng_p[:, 2], datamod_rng_p[:, 3], np.ones(len(datamod_rng_p)), datamod_rng_p[:, 1], datamod_rng_p[:, 0]], dtype=ret_dtype)
    
    ellipse_data_arr = ellipse_model(datamod_slim, ellipse_y_arr, np.asarray(datamod_rng_p[:, 4])).transpose((1, 0))
    ellipse_data_abs = np.abs(ellipse_data_arr)
    ellipse_data_index = np.argmax(np.sum(ellipse_data_abs < offset, axis=1), axis=0)
    effective_data_arr = ellipse_data_arr[ellipse_data_index]
    effective_sample_p_arr = datamod_rng_p[ellipse_data_index]
    
    return fit_rotated_ellipse(effective_data_arr, effective_sample_p_arr)


# @profile
def fit_rotated_ellipse(data, P):
    a = 1.0
    b = P[0]
    c = P[1]
    d = P[2]
    e = P[3]
    f = P[4]
    # The cost of trigonometric functions is high.
    theta = 0.5 * np.arctan(b / (a - c), dtype=np.float64)
    theta_sin = np.sin(theta, dtype=np.float64)
    theta_cos = np.cos(theta, dtype=np.float64)
    tc2 = theta_cos ** 2
    ts2 = theta_sin ** 2
    b_tcs = b * theta_cos * theta_sin
    
    # Do the calculation only once
    cxy = b ** 2 - 4 * a * c
    cx = (2 * c * d - b * e) / cxy
    cy = (2 * a * e - b * d) / cxy
    
    # I just want to clear things up around here.
    cu = a * cx ** 2 + b * cx * cy + c * cy ** 2 - f
    cu_r = np.array([(a * tc2 + b_tcs + c * ts2), (a * ts2 - b_tcs + c * tc2)])
    if cu > 1: #negatives can get thrown which cause errors, just ignore them
        wh = np.sqrt(cu / cu_r)
    else:
        pass

    w, h = wh[0], wh[1]
    
    error_sum = np.sum(data)
    # print("fitting error = %.3f" % (error_sum))
    
    return (cx, cy, w, h, theta)



TEST_VIDEO = "../assets/test.mp4"

# 320 × 240
ROI_X = 40
ROI_Y = 20
ROI_W = 260
ROI_H = 160

# Get the name and extension of the video
vid_name, vid_ext = os.path.splitext(TEST_VIDEO)

# Open the input video
input_video = cv2.VideoCapture(TEST_VIDEO)

# Get the frames per second
fps = input_video.get(cv2.CAP_PROP_FPS)

# Get the width and height of the video
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Open the output videos
output_video = cv2.VideoWriter(
    vid_name + '.out.mp4', fourcc, fps, (ROI_W, ROI_H))
morph_video = cv2.VideoWriter(
    vid_name + '.morph.mp4', fourcc, fps, (ROI_W, ROI_H), isColor=False)

# Setup pye3d
camera = CameraModel(4.0, (ROI_W, ROI_H))
detector = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)

times = []

rng = np.random.default_rng()

# Process the video
frame_number = 0
timer_video_start = time.perf_counter()
while input_video.isOpened() and output_video.isOpened():
    ret, image = input_video.read()
    if not ret:
        break

    frame_number += 1

    if frame_number < 120:
        continue

    timer_frame_start = time.perf_counter()
    timer_cv_start = timer_frame_start

    # ROI
    image = image[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    morph = 255 - closing

    timer_cv_end = time.perf_counter()

    # Debug morph video
    # morph_video.write(morph)

    timer_curves_start = time.perf_counter()

    # Contours
    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # list_of_pts = [] 
    # for ctr in contours:
        # list_of_pts += [pt[0] for pt in ctr]
    # ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)

    # Hulls
    hulls = [cv2.convexHull(contour) for contour in contours]
    # hulls = [cv2.convexHull(ctr)]

    if len(hulls) == 0: # or len(list_of_pts) == 0:
        output_video.write(image)
        continue

    # Largest hull
    largest_hull = max(hulls, key=cv2.contourArea)

    timer_curves_end = time.perf_counter()
    timer_ransac_start = timer_curves_end

    # RANSAC hull points
    try:
        cx, cy, w, h, theta = fit_rotated_ellipse_ransac(
            largest_hull.reshape(-1, 2),
            rng
        )
    except:
        # print("RANSAC failed")
        output_video.write(image)
        continue

    timer_ransac_end = time.perf_counter()

    # Draw the ellipse
    try:
        cv2.ellipse(image, (int(cx), int(cy)), (int(w), int(h)),
                    theta * 180.0 / np.pi, 0, 360, (255, 0, 0), 1)
    except:
        # print("Ellipse failed")
        # print(cx, cy, w, h, theta)
        pass

    # pye3d
    result_2d = {}
    result_2d_final = {}

    result_2d["center"] = (cx, cy)
    result_2d["axes"] = (w, h)
    result_2d["angle"] = theta * 180.0 / np.pi
    result_2d_final["ellipse"] = result_2d
    result_2d_final["diameter"] = w
    result_2d_final["location"] = (cx, cy)
    result_2d_final["confidence"] = 0.99
    result_2d_final["timestamp"] = frame_number / fps


    timer_py3d_start = time.perf_counter()

    result_3d = detector.update_and_detect(
        result_2d_final, gray, apply_refraction_correction = False
    )

    timer_py3d_end = time.perf_counter()

    # print(result_3d)
    sphere = result_3d["sphere"]
    cv2.circle(
        image,
        (
            int(sphere["center"][0]),
            int(sphere["center"][1])
        ),
        10,
        (255, 0, 255),
        1
    )

    # Draw the projected circle in red
    projected_sphere = result_3d["projected_sphere"]
    if projected_sphere is not None and "center" in projected_sphere and "axes" in projected_sphere:
        cv2.circle(
            image,
            (
                int(projected_sphere["center"][0]),
                int(projected_sphere["center"][1])
            ),
            int(abs(projected_sphere["axes"][0])),
            (0, 0, 255),
            1
        )


    # Draw the projected ellipse in green
    ellipse_3d = result_3d["ellipse"]
    if ellipse_3d is not None and "center" in ellipse_3d and "axes" in ellipse_3d and "angle" in ellipse_3d and len(ellipse_3d["center"]) == 2 and len(ellipse_3d["axes"]) == 2:
        try:
            cv2.ellipse(
                img = image,
                center = (int(ellipse_3d["center"][0]), int(ellipse_3d["center"][1])),
                axes = (int(ellipse_3d["axes"][0]), int(ellipse_3d["axes"][1])),
                angle = int(ellipse_3d["angle"]),
                startAngle = 0,
                endAngle = 360,
                color = (0, 255, 0),
                thickness = 1
            )
        except:
            # print("Failed to draw ellipse")
            # print(ellipse_3d)
            pass

    timer_frame_end = time.perf_counter()

    times.append(np.array([
        timer_frame_end - timer_frame_start,
        timer_cv_end - timer_cv_start,
        timer_curves_end - timer_curves_start,
        timer_ransac_end - timer_ransac_start,
        timer_py3d_end - timer_py3d_start
    ]))

    # Write the frame
    # output_video.write(image)

timer_video_end = time.perf_counter()
print("Total time: {:.2f} s".format(timer_video_end - timer_video_start))
print("Average FPS: {:.2f}".format(frame_number / (timer_video_end - timer_video_start)))

times_avg = np.mean(times, axis=0)
print("Average time per frame: {:.3f} ms".format(times_avg[0] * 1000))
print("    CV: {:.3f} ms".format(times_avg[1] * 1000))
print("    Curves: {:.3f} ms".format(times_avg[2] * 1000))
print("    RANSAC: {:.3f} ms".format(times_avg[3] * 1000))
print("    pye3d: {:.3f} ms".format(times_avg[4] * 1000))

# Close the videos
input_video.release()
output_video.release()
morph_video.release()
