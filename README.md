### 1. Label method

1. Static Activity

| label | Abbreviation | Description |
|-------|--------------|-------------|
|1| HandNear | hold hands near body |
|2| HandAway | hold hands away from body |
|3| HandOverHead | hands over head |
|4| PhoneLH | hold phone with left hand |
|5| PhoneRH | hold phone with right hand |
    
2. Dynamic Activity

| label | Abbreviation | Description |
|-------|--------------|-------------|
|1| Walking | Walking |
|2| Jumping | Jumping |
|3| Squating | Squating |
|4| Waving | Waving |
|5| Reaching | Reaching out for a cup |
|6| Drinking | Drinking from a cup |
|7| Boxing | Reaching out left and right arms alternatively |
|8| Bicep | Bicep curl |
|9| JumpJack | Jumping jack |

3. Attention on split method

e.g.

sequence    ##############################################

class       |-----walk-----|-----jump-----|-----wave-----|

outliers                   A              B

Original features, dist_ratio and angle_ratio might contain extreme large value (outliers):

dist_ratio = (dist[latter_framce,:] - dist[former_frame,:]) / frame_ratio
angle_ratio = (angle[latter_framce,:] - angle[former_frame,:]) / frame_ratio

Outliers usually appear during the transitional instance between 2 classes,
e.g. point A between walk and jump, point B between jump and wave

If calculate dist_ratio and angle_ratio following the formular above, outliers
will appear during the transitional instance between 2 classes, e.g.:

point A between walk and jump

point B between jump and wave

For now the solution is: calculate all features based on original data without
cutting by split method. And when define split_method.yaml, **ensure the frames
around transitional points are not included**.

e.g. walk is in frame 0-100, jump is in 100-200, wave is 200-300, (when original
data is sequential), then the split_method.yaml should be like:

Walk1:
  start: 0
  end: 98
  label: 1
Jump1:
  start: 102
  end: 198
  label: 2
Wave1:
  start: 202
  end: 300
  label: 3

In this way, the outliers would not enter final dataset for training.

This can be improved or re-written later.