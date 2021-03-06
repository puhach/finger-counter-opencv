import cv2
import numpy as np
import math

from sklearn.metrics import pairwise


# this coordinates specify the region of interest within the entire frame
roi_left = 300
roi_top = 100
roi_right = 600
roi_bottom = 400    

# this background will represent a running average of frames in the region of interest
background = None



def calc_accum_avg(frame, alpha):
    """
    Computes the running average of the frame sequence.
    
    Parameters:
        frame: An image from the sequence.
        alpha: Regulates the update speed (how fast the accumulator "forgets" about earlier images).
    """

    global background

    if background is None:
        background = frame.copy().astype("float")

    cv2.accumulateWeighted(frame, background, alpha)



def segment_region(roi, thresh):
    """
    Finds the hand contour within the region of interest.

    Parameters:
        roi: The region of interest to detect the hand within (grayscale image).
        thresh: The threshold value that is used to distinguish the hand from the background. 

    Returns:
        The thresholded b/w image and the hand contour as a tuple. 
        If no contour is found, None will be returned.
    """

    global background

    # find the difference between the current image and the background computed as an average of 
    # the initial 100 frames
    diff = cv2.absdiff(background.astype("uint8"), roi)

    # mark the hand area white againt the black background
    ret, img_bin = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    # find the external contours of the objects within the frame
    # (since opencv 3.2 the source image is not modified by this function)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) < 1:
        return None
    else:   
        # from all the contours we have found select the one with the largest area
        segment = max(contours, key=cv2.contourArea)
        return img_bin, segment



def count_fingers(bin_image, hand_segment):
    """
    Calculates the number of lifted fingers in the image.

    Parameters:
        bin_image: The thresholded frame (b/w image) within which fingers should be counted.
        hand_segment: The array of hand contour points. 

    Returns:
        The number of lifted fingers, the convex hull enclosing the hand, the array of 
        the left-most, top-most, right-most, and bottom-most vertices of the convex hull,
        the coordinates of the convex hull center (as a tuple), and the array of fingertips (points).         
    """

    # build the convex hull for the hand contour
    convex_hull = cv2.convexHull(hand_segment)

    # find the left-most, top-most, right-most, and the bottom-most points of the convex hull,
    # the convex hull seems to be in the format [[[x1, y1]],[[x2,y2]],...,[[xN,yN]]]
    left_vertex = convex_hull[convex_hull[:,:,0].argmin()][0]
    right_vertex = convex_hull[convex_hull[:,:,0].argmax()][0]
    top_vertex = convex_hull[convex_hull[:,:,1].argmin()][0]
    bottom_vertex = convex_hull[convex_hull[:,:,1].argmax()][0]

    outermost_vertices = [left_vertex, right_vertex, top_vertex, bottom_vertex]

    # calculate the center of the convex hull via the outermost vertices
    cx = (left_vertex[0] + right_vertex[0]) // 2
    cy = (top_vertex[1] + bottom_vertex[1]) // 2
    
    # calculate the distances from the center to the left-, top-, right-, 
    # and bottom-most vertices of the convex hull
    distances = pairwise.euclidean_distances(X=[[cx,cy]], Y=outermost_vertices)
    max_distance = distances.max()
    #max_distance_id = distances.argmax()
    #max_distance = distances[0,max_distance_id]
    #print(max_distance)

    radius = int(0.7*max_distance)
    circumference = 2*np.pi*radius

        
    n_fingers = 0

    # this solution (commented out) is based on the idea of counting left turns of the 
    # segments comprising the convex hull.
    """ 
    turns = []
    fingertips=[]

    if len(hand_segment) >= 3:
        u = hand_segment[len(hand_segment)-2][0]
        v = hand_segment[len(hand_segment)-1][0]

        for i in range(len(hand_segment)):
            q = hand_segment[i][0]
            uv = v - u
            vq = q - v
            d = uv[0]*vq[1] - uv[1]*vq[0]
            
            if d > 33: # significant left turn
                # not too close to the bottom
                if v[1] < cy + 0.2*(bottom_vertex[1] - cy):
                    turns.append(d)
                    fingertips.append([v])
                    n_fingers += 1

            u = v
            v = q

    fingertips=np.array(fingertips, dtype=np.int32) 
    """

    # remove vertices near the center of the hull or too close to the bottom (wrist)
    fingertips = [p for p in convex_hull 
        if math.hypot(p[0,0] - cx, p[0,1] - cy) > 0.6*max_distance 
        and p[0,1] < cy + 0.3*(bottom_vertex[1]-cy)]

    """ for p in fingertips:
        d = math.hypot(p[0,0]-cx, p[0,1]-cy)
        print(d, " ", max_distance) """

    fingertips = np.array(fingertips)

    # remove vertices which are too close to each other    
    for i in range(len(fingertips)-1):        
        j = i + 1
        while j < len(fingertips):
            x1 = fingertips[i,0,0]
            y1 = fingertips[i,0,1]
            x2 = fingertips[j,0,0]
            y2 = fingertips[j,0,1]
            #d1 = np.sqrt(np.square(x2-x1)+np.square(y2-y1))
            d = math.hypot(x2-x1, y2-y1)            
            if d < 30:
                fingertips=np.delete(fingertips, j, axis=0)                
            else:
                j += 1


    # a special heuristic to detect if there are no lifted fingers
    n_fingers = len(fingertips)
    if n_fingers > 0:
        distances = [math.hypot(p[0,0]-cx, p[0,1]-cy) for p in fingertips]
        min_distance = min(distances)
        max_distance = max(distances)
        print(max_distance, " ", min_distance)
        # small difference between max and min distance and small max distance 
        if min_distance/(max_distance+0.01) > 0.7 and max_distance < 0.4*min(bin_image.shape):
            fingertips = np.empty(shape=[0,1,2], dtype=np.int32)
            n_fingers = 0


    return n_fingers, convex_hull, np.array(outermost_vertices, dtype=np.int32).reshape(-1,1,2), (cx,cy), fingertips


def draw_convex_hull(img, convex_hull, outermost_vertices, fingertips, center, vertex_color, line_color, fingertip_color):
    """
    Draws the convex hull along with the contours and markers found in the image.

    Parameters:
        img: The image to paint in.
        convex_hull: The array of convex hull vertices.
        outermost_vertices: The array of the left-, top-, right-, and bottom-most vertices of the convex hull.
        fingertips: The array of the vertices representing fingertips detected.
        center: The tuple of (x,y) coordinates calculated from the convex hull vertices.
        vertex_color: The color for drawing the convex hull vertices.
        line_color: The color for drawing the convex hull lines.
        fingertip_color: The color for drawing the fingertip markers.
    """

    # when the list of points is in brackets it draws the lines, otherwise, only the points
    cv2.polylines(img, [convex_hull], isClosed=True, color=line_color, thickness=2)

    cv2.polylines(img
        , pts = outermost_vertices
        , isClosed=True, color=vertex_color, thickness=12)   


    #cv2.polylines(img, pts=fingertips+(roi_left,roi_top), isClosed=True, color=fingertip_color, thickness=20)
    for fingertip in fingertips:
        cv2.drawMarker(img, position=(fingertip[0,0], fingertip[0,1]), color=fingertip_color, markerType=cv2.MARKER_STAR)

    # display the convex hull center as a cross
    cv2.drawMarker(img=img, position=center, color=80, markerType=cv2.MARKER_CROSS)



cam = cv2.VideoCapture(0)
num_frames = 0

while True:

    ret, frame = cam.read()

    if not ret:
        print('Failed to read a frame.')
        break


    frame_cpy = frame.copy()
    #frame_cpy = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
    frame_roi = frame[roi_top:roi_bottom, roi_left:roi_right, :]
    frame_roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    frame_roi_gray = cv2.GaussianBlur(frame_roi_gray, ksize=(7,7), sigmaX=0)

    if num_frames < 100:  
        # first 100 frames are used to calculate the average image representing the background      
        calc_accum_avg(frame_roi_gray, 0.5)
        cv2.putText(frame_cpy, "Please wait...", org=(200,450), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, color=(255,0,0), thickness=2)        
        num_frames += 1
        #cv2.imshow('Finger Counter', frame_cpy)
    else:
        # attempt to find hand contours
        hand_ret = segment_region(frame_roi_gray, thresh=25)        

        if hand_ret is not None:
            thresholded_frame, hand_segment = hand_ret

            cv2.drawContours(frame_cpy, contours=hand_segment+(roi_left,roi_top)
                , contourIdx=-1, color=(0,180,120), thickness=5)
            
            finger_count, convex_hull, outermost_vertices, center, fingertips = count_fingers(thresholded_frame, hand_segment)

            cv2.putText(frame_cpy, str(finger_count), org=(20,20)
                , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1
                , color=(0,255,0), thickness=2)

            if len(fingertips) > 0:
                fingertips += (roi_left, roi_top)

            # draw the detection results both in the original frame and the thresholded image
            
            draw_convex_hull(frame_cpy, convex_hull+(roi_left,roi_top)
                , outermost_vertices + (roi_left, roi_top)
                , fingertips = fingertips
                , center = (center[0] + roi_left, center[1] + roi_top)
                , vertex_color=(0,0,255), line_color=(100,100,100), fingertip_color=(0,255,255))            

            draw_convex_hull(thresholded_frame, convex_hull, outermost_vertices, fingertips
                , center, vertex_color=100, line_color=50, fingertip_color=30)

                
            #mdv = outermost_vertices[mdid,0]
            #cv2.circle(img=frame_cpy, center=tuple(mdv+(roi_left,roi_top)), radius=16, color=(0,255,0), thickness=16)

            cv2.imshow('Thresholded', thresholded_frame)

    cv2.rectangle(frame_cpy, pt1=(roi_left,roi_top), pt2=(roi_right, roi_bottom), color=(255,0,0), thickness=2)

    cv2.imshow('Finger Counter', frame_cpy)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()