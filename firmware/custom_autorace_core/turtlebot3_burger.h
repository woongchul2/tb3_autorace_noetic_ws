/*******************************************************************************
* Copyright 2016 ROBOTIS CO., LTD.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/* Authors: Yoonseok Pyo, Leon Jung, Darby Lim, HanCheol Cho */

#ifndef TURTLEBOT3_BURGER_H_
#define TURTLEBOT3_BURGER_H_

#define NAME                             "Burger"

#define WHEEL_RADIUS                     0.04775f         // meter (measured diameter: 95.5 mm)
#define WHEEL_SEPARATION                 0.1482f          // meter (left/right tire center planes)
#define TURNING_RADIUS                   (WHEEL_SEPARATION / 2.0f)
#define ROBOT_RADIUS                     0.149f           // meter (conservative footprint radius: 148.645 mm)

// External spur gears are a 2:1 speed increase:
// one motor revolution produces two wheel revolutions. A single external gear
// mesh also reverses rotation. Change GEAR_DIRECTION to +1 only if the assembled
// transmission has an additional idler/stage that makes wheel and motor rotate
// in the same direction.
#define GEAR_SPEEDUP                     2.0f
#define GEAR_DIRECTION                   -1.0f
#define MOTOR_TO_WHEEL_RATIO             (GEAR_DIRECTION / GEAR_SPEEDUP)
#define MOTOR_TO_WHEEL_RATIO_ABS         (1.0f / GEAR_SPEEDUP)

// TurtleBot3MotorDriver::controlMotor() expects the wheel to be directly driven.
// Passing this equivalent radius applies MOTOR_TO_WHEEL_RATIO to motor commands
// without changing the vendor TurtleBot3 library.
#define MOTOR_COMMAND_RADIUS             (WHEEL_RADIUS / MOTOR_TO_WHEEL_RATIO)
#define ENCODER_MIN                      -2147483648     // raw
#define ENCODER_MAX                      2147483648      // raw

#define MAX_LINEAR_VELOCITY              (WHEEL_RADIUS * 2 * 3.14159265359 * 61 / 60 / MOTOR_TO_WHEEL_RATIO_ABS) // theoretical m/s at 61 motor rpm
#define MAX_ANGULAR_VELOCITY             (MAX_LINEAR_VELOCITY / TURNING_RADIUS)       // rad/s

#define MIN_LINEAR_VELOCITY              -MAX_LINEAR_VELOCITY  
#define MIN_ANGULAR_VELOCITY             -MAX_ANGULAR_VELOCITY 

#endif  //TURTLEBOT3_BURGER_H_
