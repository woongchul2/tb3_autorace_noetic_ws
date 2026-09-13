std_msgs/Header header

uint8 NONE=0
uint8 INTERSECTION_WARNING=1
uint8 CONSTRUCTION_WARNING=2
uint8 TUNNEL_WARNING=3
uint8 PARKING=4
uint8 DIRECTION=5
uint8 LEVEL_CROSSING_WARNING=6
uint8 UNKNOWN_WARNING=255

uint8 DIRECTION_NONE=0
uint8 DIRECTION_LEFT=1
uint8 DIRECTION_RIGHT=2

uint8 sign_type
uint8 direction
float32 confidence
sensor_msgs/RegionOfInterest roi
