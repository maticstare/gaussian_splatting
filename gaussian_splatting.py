import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_splats(filename):
    # Load splats from binary file
    # 3x4B position, 3x4B scale, 4x1B color(RGBA), 4x1B rotation(xyzw)
    # 12B + 12B + 4B + 4B = 32B per splat
    with open(filename, 'rb') as f:
        data = f.read()
    num_splats = len(data) // 32
    splatsFloat = np.frombuffer(data, dtype=np.float32).reshape(num_splats, 8)
    splatsInt = np.frombuffer(data, dtype=np.uint8).reshape(num_splats, 32)

    splats = np.zeros((len(splatsInt),14))
    for i in range(len(splatsInt)):
        quaternion = (np.array(splatsInt[i,28:], dtype=np.int16)-128)/128
        splats[i] = np.append(np.append(splatsFloat[i,:6], splatsInt[i,24:28]), quaternion).flatten()
    
    return splats


def plot_splats_sample(splats, sample_size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = splats[:,0]
    y = splats[:,1]
    z = splats[:,2]

    s1 = splats[:,3]
    s2 = splats[:,4]
    s3 = splats[:,5]

    S = np.dstack([s1,s2,s3])[0]
    r, g, b, a = splats[:,6], splats[:,7], splats[:,8], splats[:,9]
    
    C = np.dstack([r,g,b,a])[0]
        #print(C)

    xyz = np.dstack([x,y,z])[0]
    idx = np.random.randint(len(x), size=sample_size)
    sample = xyz[idx, :]

    """ ax.scatter(sample[:,0], sample[:,1], sample[:,2], c=C[idx,:]) """

    ax.scatter(sample[:,0], sample[:,1], sample[:,2], s=S[idx,0]*10, c=C[idx,0])
    ax.scatter(sample[:,0], sample[:,1], sample[:,2], s=S[idx,1]*10, c=C[idx,1])
    ax.scatter(sample[:,0], sample[:,1], sample[:,2], s=S[idx,2]*10, c=C[idx,2])
    
    plt.show()
    r1 = (splats[:,10]+128)/255
    r2 = (splats[:,11]+128)/255
    r3 = (splats[:,12]+128)/255
    r4 = (splats[:,13]+128)/255
    #print(r1.max(), r2.max(), r3.max(), r4.max())
    #print(r1.min(), r2.min(), r3.min(), r4.min())


def project_pointcloud(splats):
    # Define intrinsic camera parameters 
    focal_length = 150
    image_width = 1920
    image_height = 1080
    intrinsic_matrix = np.array([ 
        [focal_length, 0, image_width/2], 
        [0, focal_length, image_height/2], 
        [0, 0, 1] 
    ]) 

    # Define extrinsic camera parameters 
    rvec = np.array([0, 0, 0], dtype=np.float32) 
    tvec = np.array([0, 0, 0], dtype=np.float32) 

    points_3d = np.dstack([splats[:,0],splats[:,1],splats[:,2]])[0]
    
    # Calculate distances from points to the camera
    # sorted only by z because the camera is looking in the z direction
    distances = np.array(np.abs(points_3d[:, 2] - tvec[2]))
    #print(distances.max(), distances.min())

    # Sort points and colors by distance in descending order
    sorted_indices = np.argsort(-distances)
    points_3d = points_3d[sorted_indices]
    splats = splats[sorted_indices]

    # Project 3D points onto 2D plane 
    points_2d, _ = cv2.projectPoints(points_3d, 
                                    rvec,
                                    tvec.reshape(-1, 1), 
                                    intrinsic_matrix, 
                                    None) 
    points_2d = points_2d.astype(int)

    # Generate 2d points matrix [(x, y, b, g, r, a)]
    points_pos_color_2d = np.zeros((len(points_2d),6))
    r, g, b, a = splats[:,6], splats[:,7], splats[:,8], splats[:,9]

    for i in range(len(points_pos_color_2d)):
        points_pos_color_2d[i] = np.array([points_2d[i][0][0], points_2d[i][0][1], b[i], g[i], r[i], a[i]])

    # Apply Gaussian falloff
    #points_pos_color_2d = apply_gaussian_falloff(points_pos_color_2d, distances, s)
    

    return points_pos_color_2d.astype(int), distances



def apply_gaussian_falloff(c, x, s, z, a):
    x_minus_c = np.subtract(x, c)    
    x_minus_c_transpose = np.transpose(x_minus_c)
    z_over_s = z / s
    sigma_inv = np.array([[z_over_s, 0, 0], [0, z_over_s, 0], [0, 0, z_over_s]])

    g_x = np.exp(-0.5 * np.matmul(np.matmul(x_minus_c_transpose, sigma_inv), x_minus_c))
    return g_x * a

def draw_2d_image(xybgra, distances, s):
    image_height = 1080
    image_width = 1920
    # Plot 2D points 
    img = np.ones((image_height, image_width, 3), dtype=np.float32)
    max_distance = distances.max()
    
    for i in range(len(xybgra)):
        x, y, b, g, r, a = xybgra[i].tolist()
        a /= 255
        # Scale size based on distance
        #size = round((((2 * s * max_distance) / distances[i])*10) % 10)
        size = int((2 * s * max_distance) / distances[i])

        top_left = (max(0, x - size), max(0, y - size))
        bottom_right = (min(image_width - 1, x + size), min(image_height - 1, y + size))
        # Apply alpha blending
        img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
            (1 - a) * img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] + \
            a * (np.array([b, g, r])/255)
        
        """ for y1 in range(top_left[1], bottom_right[1]):
            for x1 in range(top_left[0], bottom_right[0]):
                a_adjusted = apply_gaussian_falloff((x, y, 0), (x1, y1, 0), s, distances[i], a)
                img[y1, x1] = (1 - a_adjusted) * img[y1, x1] + a_adjusted * (np.array([b, g, r])/255) """

    cv2.imshow('Image', img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()



if __name__ == "__main__":
    splats = load_splats("splats/train.splat")
    #plot_splats_sample(splats, len(splats)//1)

    s = 0.5
    xybgra, distances = project_pointcloud(splats)
    draw_2d_image(xybgra, distances, s)