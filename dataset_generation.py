import numpy as np

def get_coords(path):
    Path = get_path()
    csv_df = pd.read_csv(Path,header=None) # ,header=None
    csv_np = csv_df.to_numpy()
    # center of gravity
    cogs = np.expand_dims(csv_np[:,0:3],axis=1)
    # coordinates of all joints
    x_coord = np.expand_dims(csv_np[:,3::3], axis=1)
    y_coord = np.expand_dims(csv_np[:,4::3], axis=1)
    z_coord = np.expand_dims(csv_np[:,5::3], axis=1)
    coords = np.concatenate((x_coord,y_coord,z_coord), axis=1)
    return coords,cogs

if __name__ == '__main__':
    path = ''
    coords,cogs = get_coords(path)
    print(f'Coordinates shape: {coords.shape}')
    print(f'Center of gravity shape: {cogs.shape}')