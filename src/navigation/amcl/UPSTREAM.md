# Vendored AMCL

This package is the ROS Navigation `amcl` package from tag `1.17.3`, commit
`f13af47ee2bca6c3bf99db2965e46e55303f6e66`:

<https://github.com/ros-planning/navigation/tree/1.17.3/amcl>

The workspace copy adds the opt-in `diff-signed` odometry model. It preserves
the longitudinal sign of differential-drive motion below the stock model's
1 cm bearing guard. The original `diff`, `omni`, `diff-corrected`, and
`omni-corrected` behaviours remain unchanged.
