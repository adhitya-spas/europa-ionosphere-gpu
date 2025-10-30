#Generate HF ray paths from scratch
import numpy as np
import matplotlib.pyplot as plt

# Let ionosphere be 0-1500km and latitude 0deg to 10 deg
ionosphere_height = (0, 1500)  # in km
latitude_range = (0, 10)  # in degrees

# introduce angles - spacecraft HF pointing angles are the same as the incidence angles of the rays
angle_of_incidence = [1,2,3,4,5]#[5, 15, 25, 35, 45]

#Start from 1500 km altitude at 0deg lat. and trace rays down to 0 km altitude at each incidence angle
start_altitude = 1500  # in km
end_altitude = 0  # in km
start_latitude = 0  # in degrees
end_latitude = 10  # in degrees
ray_paths = []

for angle in angle_of_incidence:
    # Trace the ray path for each angle
    ray_path = []
    reflected_ray_path = []
    
    # Get the horizontal component of the ray path
    horizontal_distance = 1500 * np.tan(np.deg2rad(angle))  # in km

    # Get points in between the ray path
    number_of_pts = 100
    ys = np.linspace(start_altitude, end_altitude, num=number_of_pts)
    xs = np.linspace(start_latitude, start_latitude + horizontal_distance, num=number_of_pts)

    # Convert the km to deg for latitude (approximation) for Europa
    latitude = xs / (2 * np.pi * 1560.8) * 360

    # maybe later crop to end_latitude

    # Trace reflected ray path
    horizontal_distance_reflected = 1500 * np.tan(np.deg2rad(angle))  # in km

    # Get points in between the ray path
    number_of_pts = 100
    ys_ref = np.linspace(end_altitude, start_altitude, num=number_of_pts)
    xs_ref = np.linspace(start_latitude + horizontal_distance, start_latitude + horizontal_distance + horizontal_distance_reflected, num=number_of_pts)

    # Convert the km to deg for latitude (approximation) for Europa
    latitude_ref = xs_ref / (2 * np.pi * 1560.8) * 360

    # Store the all ray paths as multiple arrays in one list
    ray_path = np.column_stack((ys, latitude))
    reflected_ray_path = np.column_stack((ys_ref, latitude_ref))
    ray_paths.append(ray_path)
    ray_paths.append(reflected_ray_path)

# Now plot all those ray paths

plt.figure(figsize=(10, 6))
for i, ray_path in enumerate(ray_paths):
    plt.plot(ray_path[:, 1], ray_path[:, 0], label=f"Angle: {angle_of_incidence[i // 2]}°")
plt.ylim(0, start_altitude)
plt.xlim(0, end_latitude)
plt.xlabel("Latitude (degrees)")
plt.ylabel("Altitude (km)")
plt.title("HF Ray Paths through the Ionosphere")
plt.legend()
plt.grid()
plt.savefig("HF_Ray_Paths.png", dpi=300)