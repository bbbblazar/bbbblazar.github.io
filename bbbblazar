import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from moviepy.editor import VideoFileClip

def getCameraCalibrationCoefficients(chessboardname, nx, ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(chessboardname)
    if len(images) > 0:
        print("images num for calibration : ", len(images))
    else:
        print("No image for calibration.")
        return

    ret_count = 0
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (img.shape[1], img.shape[0])
        # Finde the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            ret_count += 1
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print('Do calibration successfully')
    return ret, mtx, dist, rvecs, tvecs
def undistortImage(distortImage, mtx, dist):
    return cv2.undistort(distortImage, mtx, dist, None, mtx)

def get_M_Minv_img(img):
    src = np.float32([[203, 720], [585, 460], [695, 460], [1127, 720]])
    dst = np.float32([[320, 720], [320, 0], [960, 0], [960, 720]])
    h,w=img.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img = cv2.warpPerspective(img,M, (w, h), flags=cv2.INTER_LINEAR)
    return img,M, Minv
#对l通道进行提取获得白色车道线
def hlsLSelect(img,thresh=(220,255)):
    h,w=img.shape[:2]
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L_channel = hls[:, :, 1]
    L_channel = L_channel * (255 / np.max(L_channel))
    binary_output = np.zeros_like(L_channel)
    binary_output[(L_channel > thresh[0]) & (L_channel <= thresh[1])] = 1
    return binary_output

#对b通道进行提取获得白色车道线
def labBSelect(img, thresh=(195, 255)):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_b = lab[:,:,2]
    if np.max(lab_b) > 100:
        lab_b = lab_b*(255/np.max(lab_b))
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    return binary_output
#合并图像
def combinary_l_and_b(lab,hls):
    out=np.zeros_like(lab)
    h,w=lab.shape[:2]
    for i in range(h):
        for j in range(w):
            if lab[i][j] == 1 or hls[i][j] == 1:
                out[i][j] = 1
    return out


def cal_line_param(binary_warped):
    # 1.确定左右车道线的位置
    # 统计直方图
    histogram = np.sum(binary_warped[:, :], axis=0)
    # 在统计结果中找到左右最大的点的位置，作为左右车道检测的开始点
    # 将统计结果一分为二，划分为左右两个部分，分别定位峰值位置，即为两条车道的搜索位置
    midpoint = np.int32(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # 2.滑动窗口检测车道线
    # 设置滑动窗口的数量，计算每一个窗口的高度
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    # 获取图像中不为0的点
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # 车道检测的当前位置
    leftx_current = leftx_base
    rightx_current = rightx_base
    # 设置x的检测范围，滑动窗口的宽度的一半，手动指定
    margin = 100
    # 设置最小像素点，阈值用于统计滑动窗口区域内的非零像素个数，小于50的窗口不对x的中心值进行更新
    minpix = 50
    # 用来记录搜索窗口中非零点在nonzeroy和nonzerox中的索引
    left_lane_inds = []
    right_lane_inds = []

    # 遍历该副图像中的每一个窗口
    for window in range(nwindows):
        # 设置窗口的y的检测范围，因为图像是（行列）,shape[0]表示y方向的结果，上面是0
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # 左车道x的范围
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        # 右车道x的范围
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # 确定非零点的位置x,y是否在搜索窗口中，将在搜索窗口内的x,y的索引存入left_lane_inds和right_lane_inds中
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 如果获取的点的个数大于最小个数，则利用其更新滑动窗口在x轴的位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # 将检测出的左右车道点转换为array
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 获取检测出的左右车道点在图像中的位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 3.用曲线拟合检测出的点,二次多项式拟合，返回的结果是系数
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

#def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50):
    #lefttx,leftty,righttx,rightty,out_img=find_lane_pixels(binary_warped, nwindows, margin, minpix)
    #left_fit=np.polyfit(lefttx,leftty,2)
    #right_fit=np.polyfit(righttx,rightty,2)
    #ploty=np.linspace(0,binary_warped.shape[0]-1,binary_warped.shape[0])
    #left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    #right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


    #out_img[leftty, lefttx] = [255, 0, 0]
    #out_img[rightty, righttx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #return out_img,left_fit,right_fit

def calculate_curv_and_pos(binary_warped,left_fit, right_fit):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)
    left_cur_fit=np.polyfit(ploty*ym_per_pix,leftx*xm_per_pix,2)
    right_cur_fit=np.polyfit(ploty*ym_per_pix,rightx*xm_per_pix,2)

    left_curverad=((1 + (2*left_cur_fit[0]*y_eval*ym_per_pix + left_cur_fit[1])**2)**1.5) / np.absolute(2*left_cur_fit[0])
    right_curverad=((1 + (2*right_cur_fit[0]*y_eval*ym_per_pix + right_cur_fit[1])**2)**1.5) / np.absolute(2*right_cur_fit[0])
    curverad=(left_curverad+right_curverad)/2.
    #
    lane_width=np.absolute(leftx[719]-rightx[719])
    lane_xm_per_pix=3.7/lane_width
    veh_pos=(leftx[719]+rightx[719])*lane_xm_per_pix/2.
    cen_pos=binary_warped.shape[1]*lane_xm_per_pix/2.
    distance_from_center=np.absolute(cen_pos-veh_pos)
    return  distance_from_center,curverad

def draw_area(undist,binary_warped,Minv,left_fit, right_fit):
    ploty=np.linspace(0,binary_warped.shape[0]-1,binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp=np.dstack((warp_zero,warp_zero,warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp,np.int_([pts]),(0,255,0))

    newwarp=cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    return result

def draw_values(img, curvature, distance_from_center):
    font = cv2.FONT_HERSHEY_SIMPLEX
    radius_text = "Radius of Curvature: %sm" % (round(curvature))

    #if distance_from_center > 0:
        #pos_flag = 'right'
    #else:
        #pos_flag = 'left'

    cv2.putText(img, radius_text, (100, 100), font, 1, (255, 255, 255), 2)
    #center_text = "Vehicle is %.3fm %s of center" % (abs(distance_from_center), pos_flag)
    center_text = "Vehicle offset lane center is: %.3fm " % (distance_from_center)
    cv2.putText(img, center_text, (100, 150), font, 1, (255, 255, 255), 2)
    cv2.putText(img,"learner:chenJJ", (100, 200), font, 1, (255, 255, 255), 2)
    return img

nx = 9
ny = 6
video_input = 'project_video.mp4'
video_output = 'result_video.mp4'
fourcc = cv2.VideoWriter_fourcc('M','P','4','V')
out = cv2.VideoWriter(video_output, fourcc, 20.0, (1280, 720))
ret, mtx, dist, rvecs, tvecs = getCameraCalibrationCoefficients('../IR_camera_calib_img/calibration*.jpg', nx, ny)
cap = cv2.VideoCapture(video_input)
while True:
    bet, image = cap.read()
    undistort_image = undistortImage(image, mtx, dist)
    img, M, Minv = get_M_Minv_img(undistort_image)
    # imgorigin = cv2.warpPerspective(img,Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    hls_l = hlsLSelect(img, thresh=(220, 255))
    lab_b = labBSelect(img, thresh=(195, 255))
    combined_binary = combinary_l_and_b(lab_b, hls_l)
    left_fit, right_fit = cal_line_param(combined_binary)
    # out_img=fill_lane_poly(img, left_fit, right_fit)
    distance_from_center, curverad= calculate_curv_and_pos(combined_binary,left_fit, right_fit)
    result = draw_area(undistort_image, combined_binary, Minv, left_fit, right_fit)
    result=draw_values(result, curverad, distance_from_center)
    cv2.imshow("Frame",result)
    out.write(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头 release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()
