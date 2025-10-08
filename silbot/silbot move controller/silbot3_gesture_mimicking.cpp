/*
 * silbot3_tele_keyboard.cpp
 *
 *  Created on: 2015. 11. 9.
 *      Author: zikprid
 */

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <signal.h>
#include <string.h>
#include <unistd.h> // for usleep() only
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <map>
#include <math.h>
#include <fstream>

#include <X11/Xlib.h>
#include <X11/X.h>

#include <string.h>
#include <ros/ros.h>
#include <silbot3_msgs/Device_Wheel_Msg.h>
#include <silbot3_msgs/Device_Arm_Msg.h>
#include <silbot3_msgs/Device_ErobotPantilt_Msg.h>
#include <std_msgs/String.h>
#include <std_msgs/Int32.h>

// #include <std_msgs/Bool.h>
#include <cstring>

using namespace std;

#define VERSION         "0.1"
#define DEFAULT_DELAY   10000
#define BIT(c, x)       ( c[x/8]&(1<<(x%8)) )
#define TRUE            1
#define FALSE           0

#define KEYSYM_STRLEN   64

#define SHIFT_DOWN      1
#define LOCK_DOWN       5
#define CONTROL_DOWN    3
#define ISO3_DOWN       4
#define MODE_DOWN       5

/* It is pretty standard */
#define SHIFT_INDEX     1  /*index for XKeycodeToKeySym(), for shifted keys*/
#define MODE_INDEX      2
#define MODESHIFT_INDEX 3
#define ISO3_INDEX      4
#define ISO3SHIFT_INDEX 4

#define _USE_MATH_DEFINES


struct Joint_values 
{
    const int max_speed;
    const double min_bound;
    const double max_bound;
};

map<string, Joint_values> joint_information;
vector<string> joint_names;

void map_joint_info() {
    Joint_values jv1 = {150, -20.0, 90.0};
    Joint_values jv2 = {150, -45.0, 90.0};
    Joint_values jv3 = {150, -35.0, 35.0};
    Joint_values jv4 = {150, -90.0, 180.0};
    
    joint_information.insert(make_pair("LShoulderRoll"      , jv1));
    joint_information.insert(make_pair("LElbowRoll"         , jv2));
    joint_information.insert(make_pair("RShoulderRoll"      , jv1));
    joint_information.insert(make_pair("RElbowRoll"         , jv2));
    joint_information.insert(make_pair("HeadYaw"            , jv3));
    joint_information.insert(make_pair("HeadPitch"          , jv3));
    joint_information.insert(make_pair("LShoulderPitch"     , jv4));
    joint_information.insert(make_pair("RShoulderPitch"     , jv4));

    joint_names.push_back("LShoulderRoll");
    joint_names.push_back("LElbowRoll");
    joint_names.push_back("RShoulderRoll");
    joint_names.push_back("RElbowRoll");
    joint_names.push_back("HeadYaw");
    joint_names.push_back("HeadPitch");
    joint_names.push_back("LShoulderPitch");
    joint_names.push_back("RShoulderPitch");
}

// Limit the movement and speed of movements 
vector<vector<double> > move_limiter(vector<vector<double> > data, vector<string> name_arr, vector<double> time_arr, int size)
{
    string name;
    map<string, Joint_values>::iterator ji_iter;

    for(int i = 0; i < 8; i++) {

        ji_iter = joint_information.find(name_arr[i]);

        if (data[0][i] < ji_iter->second.min_bound) { data[0][i] = ji_iter->second.min_bound; }
        if (data[0][i] > ji_iter->second.max_bound) { data[0][i] = ji_iter->second.max_bound; }


        for(int j = 1; j < size; j++) {

            // If joint is moving too fast, reassign angles by an acceptable amount
            if ((data[j][i] > data[j-1][i]) && ((data[j][i] - data[j-1][i])/(time_arr[i] - time_arr[i-1]) > (ji_iter->second.max_speed))) {
                data[j][i] = data[j-1][i] + ((time_arr[i] - time_arr[i-1]) * (ji_iter->second.max_speed));
            } else if ((data[j][i] < data[j-1][i]) && ((data[j][i] - data[j-1][i])/(time_arr[i] - time_arr[i-1]) < -(ji_iter->second.max_speed))) {
                data[j][i] = data[j-1][i] - ((time_arr[i] - time_arr[i-1]) * (ji_iter->second.max_speed));
            }
    
            // If joint is outside of range, reassign to min/max range bounds
            if (data[j][i] < ji_iter->second.min_bound) { data[j][i] = ji_iter->second.min_bound; }
            if (data[j][i] > ji_iter->second.max_bound) { data[j][i] = ji_iter->second.max_bound; }
            
            // // Ensure the arms do not collide with body or head
            // if ((ji_iter->first == "LElbowRoll")) {
            //     if (data[j][6] >= -50 && data[j][6] < 65 && data[j][0] < 35) {
            //         if (data[j][i] < (-(8 * data[j][0]) / 5) - 22) {
            //             data[j][i] = (-(8 * data[j][0]) / 5) - 22;
            //             cout << "Collision for LElbow @ " << j << ", " << i << '\n';
            //         }
            //     } else if (data[j][6] >= 125 && data[j][0] < 40) {
            //         if (data[j][i] < (-(8 * data[j][0]) / 5) - 15) {
            //             data[j][i] = (-(8 * data[j][0]) / 5) - 15;
            //             cout << "Collision for LElbow @ " << j << ", " << i << '\n';
            //         }
            //     } else if (data[j][0] < 0) {
            //         if (data[j][i] < (-(4 * data[j][0]) / 3) - 76) {
            //             data[j][i] = (-(4 * data[j][0]) / 3) - 76;
            //             cout << "Collision for LElbow @ " << j << ", " << i << '\n';
            //         }
            //     }
            // }

            // if ((ji_iter->first == "RElbowRoll")) {
            //     if (data[j][7] >= -50 && data[j][7] < 65 && data[j][2] < 35) {
            //         if (data[j][i] < (-(8 * data[j][2]) / 5) - 22) {
            //             data[j][i] = (-(8 * data[j][2]) / 5) - 22;
            //             cout << "Collision for RElbow @ " << j << ", " << i << '\n';
            //         }
            //     } else if (data[j][7] >= 125 && data[j][2] < 40) {
            //         if (data[j][i] < (-(8 * data[j][2]) / 5) - 15) {
            //             data[j][i] = (-(8 * data[j][2]) / 5) - 15;
            //             cout << "Collision for RElbow @ " << j << ", " << i << '\n';
            //         }
            //     } else if (data[j][2] < 0){
            //         if (data[j][i] < (-(4 * data[j][2]) / 3) - 76) {
            //             data[j][i] = (-(4 * data[j][2]) / 3) - 76;
            //             cout << "Collision for RElbow @ " << j << ", " << i << '\n';
            //         }
            //     }
            // }
        }
    }
    return data;
}

/**
    Helper function to convert strings to integers
*/
int stoi_c(string s)
{
    int i;
    int inv = 1;
    i = 0;
    
    // Loop through char array until irrelevant char is found
    for(int k = 0; k < s.size(); k++)
    {
        if (s[k] >= '0' && s[k] <= '9') {
            i = i * 10.0 + (s[k] - '0');  // Shift numbers across and numerically append newest char to number
            
        // Set result to negative if '-' is found 
        } else if (s[k] == '-') {
            inv = -1;
        
        // Immediately return if a decimal point is found
        } else if (s[k] == '.') {
            return inv * i;
        }
    }

    return inv * i;
}

/**
    Helper function to convert strings to doubles
*/
double stod_c(string s)
{
    double i, inv = 1.0;
    int dec = 0, dp = 0;
    i = 0;

    
    // Loop through char array until irrelevant char is 
    // found or current decimal place exceeds the max decimal place
    for(int k = 0; k < s.size(); k++)
    {
        if (s[k] >= '0' && s[k] <= '9') {
            i = i * 10.0 + (s[k] - '0');  // Shift numbers across and numerically append newest char to number
            dec *= 10;                  // Increase divisor if decimal point has been found
            if (dec > 0) {              // Track decimal point count
                dp++;
            }
            
        // Begin tracking decimal place
        } else if (s[k] == '.') {
            dec = 1;
            
        // Set result to negative if '-' is found 
        } else if (s[k] == '-') {
            inv = -1.0;
        }

        // Prevent result from going beyond 2 decimal places
        if (dp >= 2) {
            break;
        }
    }
    
    if (dec == 0) { dec = 1; }  // Ensure no division by zero occurs
    return (inv * i)/dec;
}

int pan = 0;
int tilt = 0;
int xtion = 0;
int xySpeed = 300;
int tSpeed = 40;

ros::Publisher armPublisher;
ros::Publisher pantiltPublisher;
ros::Publisher gesturePublisher;

ros::Subscriber gptSubscriber;
ros::Subscriber faceSubscriber;
ros::Subscriber shitSubscriber;


void moveArmWithSpeed(int angle0, int angle1, int angle2, int angle3, int angle4, int angle5, int speed0, int speed1, int speed2, int speed3, int speed4, int speed5)
{
    silbot3_msgs::Device_Arm_Msg msg;
    msg.command = "ARM_MOVE_TO_POSITION_ALL_BOTH_WITH_AXIS_SPEED";
    msg.angles.push_back(angle0);
    msg.angles.push_back(angle1);
    msg.angles.push_back(angle2);
    msg.angles.push_back(angle3);
    msg.angles.push_back(angle4);
    msg.angles.push_back(angle5);
    msg.speeds.push_back(speed0);
    msg.speeds.push_back(speed1);
    msg.speeds.push_back(speed2);
    msg.speeds.push_back(speed3);
    msg.speeds.push_back(speed4);
    msg.speeds.push_back(speed5);
    armPublisher.publish(msg);
    }

void movePantilt(double pan, double tilt, double xtion, int speedPantilt, int speedXtion)
{
	silbot3_msgs::Device_ErobotPantilt_Msg msg;
	msg.command = "PANTILT_MOVE_ABSOLUTE_POSITION_ALL";
	msg.angles.push_back(pan);
	msg.angles.push_back(tilt);
	msg.angles.push_back(xtion);
	msg.speeds.push_back(speedPantilt);
	msg.speeds.push_back(speedPantilt);
	msg.speeds.push_back(speedXtion);

	pantiltPublisher.publish(msg);
}   

void moveArm(int angle0, int angle1, int angle2, int angle3, int angle4, int angle5, int speed)
{
	silbot3_msgs::Device_Arm_Msg msg;
	msg.command = "ARM_MOVE_TO_POSITION_ALL_BOTH";
	msg.angles.push_back(angle0);
	msg.angles.push_back(angle1);
	msg.angles.push_back(angle2);
	msg.angles.push_back(angle3);
	msg.angles.push_back(angle4);
	msg.angles.push_back(angle5);
	msg.speeds.push_back(speed);
	armPublisher.publish(msg);
}


int face_detected = 0;


// Gesture variables
vector<vector<vector<int> > *> gvec, svec;
vector<vector<double> *> tvec;
vector<vector<int> > wave_gesture, celebrate_gesture, think_gesture, shake_gesture, thunk_gesture, wave_speed, celebrate_speed, think_speed, shake_speed, thunk_speed;
vector<double> wave_time, celebrate_time, think_time, shake_time, thunk_time;
vector<int> recent_action;

// Reads gestures into memory
void initMotions(float time_fac)
{
    time_t start_time, curr_time;
    fstream fin;
    int time_var = 2, head = 0;
    double time_mult = 1.0;
    std_msgs::Int32 msg_pub;
    for (int i = 0; i < 10; i++) {
        recent_action.push_back(0);
    }

    gvec.push_back(&wave_gesture);
    gvec.push_back(&celebrate_gesture);
    gvec.push_back(&think_gesture);
    gvec.push_back(&shake_gesture);
    gvec.push_back(&thunk_gesture);

    svec.push_back(&wave_speed);
    svec.push_back(&celebrate_speed);
    svec.push_back(&think_speed);
    svec.push_back(&shake_speed);
    svec.push_back(&thunk_speed);

    tvec.push_back(&wave_time);
    tvec.push_back(&celebrate_time);
    tvec.push_back(&think_time);
    tvec.push_back(&shake_time);
    tvec.push_back(&thunk_time);


    map_joint_info();

    // Motion numbers:
    //      - 0: Wave
    //      - 1: Think
    //      - 2: Head Shake
    //      - 3: Think Finish
    for(int ges = 0; ges < 5; ges++) {
        vector<vector<double> > content, angle_arr, test_angle_arr;
        vector<vector<int> > angle_input, speed;
        vector<double> row, time_arr;
        vector<int> empty_vector;
        string line, word;
        int time_col = 0;

        float smooth_factor = 1.2;
        float speed_factor = 1.0;


        // if (ges == 0) { fin.open("/home/silbot3/gestures/wave_silbot.csv"); }
        if (ges == 0) { fin.open("/home/silbot3/gestures/wave.csv"); }
        // else if (ges == 1) { fin.open("/home/silbot3/gestures/celebrate.csv"); }
        else if (ges == 1) { fin.open("/home/silbot3/gestures/thinking.csv"); }
        else if (ges == 2) { fin.open("/home/silbot3/gestures/big-wave.csv"); }
        // else if (ges == 3) { fin.open("/home/silbot3/gestures/thinking.csv"); }
        // else if (ges == 2) { fin.open("/home/silbot3/gestures/celebrate_silbot.csv"); }
        else if (ges == 3) { fin.open("/home/silbot3/gestures/think-finish.csv"); }
        else if (ges == 4) { fin.open("/home/silbot3/gestures/celebrate.csv"); }


        // Read gesture CSV
        if(fin.is_open()) {
            while (getline(fin, line)) {
            istringstream s(line);

            row.clear();

            while(getline(s, word, ','))
            {
                
                if(time_col == 0)
                {
                    row.push_back(stod_c(word));
                    time_col++;
                } else {
                    row.push_back(stod_c(word) * (180.0/M_PI));     // Convert radians to degrees
                }
            }
            content.push_back(row);
            time_col = 0;
            }
        } else {
            cout<<"Could not open the file\n";
        }

        // Split CSV into separate time and angle data
        for(int j = 0; j < content.size(); j++)
        {
            time_arr.push_back(content[j][0]);
            angle_arr.push_back(vector<double> (content[j].begin() + 1, content[j].end() - 2));     // Ignore Time, HipRoll, and HipPitch columns
        }

        // Compatibility translation
        for (int i = 1; i < angle_arr.size(); i++) {
            // Invert LElbowRoll, RShoulderRoll, and LElbowYaw angles
            angle_arr[i][1] = -1 * (angle_arr[i][1]);
            angle_arr[i][2] = -1 * (angle_arr[i][2]);
            angle_arr[i][8] = -1 * (angle_arr[i][8]);
            
            // Shift angles by pi/2
            angle_arr[i][6] = 90 - (angle_arr[i][6]);
            angle_arr[i][7] = 90 - (angle_arr[i][7]);   
            
            // LShoulderPitch = LShoulderPitch + LElbowYaw
            angle_arr[i][6] = angle_arr[i][6] + angle_arr[i][8];
            if (angle_arr[i][6] < 0) {
                angle_arr[i][6] = 0;
            }
            
            // RShoulderPitch = RShoulderPitch + RElbowYaw
            angle_arr[i][7] = angle_arr[i][7] + angle_arr[i][9];
            if (angle_arr[i][7] < 0) {
                angle_arr[i][7] = 0;
            }
        }

        angle_arr = move_limiter(angle_arr, joint_names, time_arr, time_arr.size());

        // Convert each angle to type int
        for(int mm_iter1 = 0; mm_iter1 < angle_arr.size(); mm_iter1++) {
            empty_vector.clear();
            angle_input.push_back(empty_vector);
            for(int mm_iter2 = 0; mm_iter2 < angle_arr[0].size(); mm_iter2++) {
                angle_input[mm_iter1].push_back((int)(angle_arr[mm_iter1][mm_iter2]));
            }
        }
        
        (*gvec[ges]) = angle_input;

        // Set variable joint speeds based on required distance to cover
        empty_vector.clear();
        speed.push_back(empty_vector);
        for(int j = 0; j < angle_arr[0].size(); j++) {
            speed[0].push_back(60);
        }

        for(int k = 1; k < angle_arr.size(); k++) {
            empty_vector.clear();
            speed.push_back(empty_vector);
            for(int j = 0; j < angle_arr[k].size(); j++) {
                speed[k].push_back(abs((angle_arr[k][j] - angle_arr[k-1][j]) / (time_arr[k] - time_arr[k-1])) / smooth_factor);
            }
        }
        (*svec[ges]) = speed;
        
        // Increase the speed of motions
        for (int t = 0; t < time_arr.size(); t++) {
            time_arr[t] = time_arr[t]/speed_factor;
        }
        (*tvec[ges]) = time_arr;

        fin.close();
    }
}

void applyMotions(string motion)
{
    vector<vector<int> > angle_input, speed;
    vector<double> time_arr;
    std_msgs::Int32 msg_pub;
    time_t start_time, curr_time;
	start_time = time(0);
	curr_time = time(0);
    
    if ((motion.compare("wave") == 0) || (motion.compare("0") == 0))                                { angle_input = *gvec[0]; speed = *svec[0]; time_arr = *tvec[0]; }
    else if ((motion.compare("Think") == 0) || (motion.compare("1") == 0))                               { angle_input = *gvec[1]; speed = *svec[1]; time_arr = *tvec[1]; }
    else if ((motion.compare("good") == 0 || motion.compare("bad") == 0 || motion.compare("2") == 0))    { angle_input = *gvec[2]; speed = *svec[2]; time_arr = *tvec[2]; }
    else if ((motion.compare("Think_finish") == 0) || (motion.compare("3") == 0))                        { angle_input = *gvec[3]; speed = *svec[3]; time_arr = *tvec[3]; }
    else if ((motion.compare("Thinking") == 0) || (motion.compare("4") == 0))                            { angle_input = *gvec[4]; speed = *svec[4]; time_arr = *tvec[4]; }
    else { cout << "No match" << '\n'; return; } 
    // time_arr[0] = time_arr[1]/2.0;

    // // add the current predicted joint data to the beginning of the motion
    // for (int i = 0; i < recent_action.size(); i++) {
    //     angle_input[i].insert(angle_input[i].begin(), recent_action[i]);
    //     speed[i].insert(speed[i].begin(), abs((angle_input[i][1] - angle_input[i][0])/(time_arr[1]- time_arr[0])));
    // }
    // cout << recent_action.size() << "\n (";
    // for (int i = 0; i < recent_action.size(); i++) {
    //     cout << recent_action[i] << ", ";
    // }
    // cout << ") \n";


	// Apply joint angles across predefined time sequence
    if ((motion.compare("wave") == 0) || (motion.compare("Think") == 0) || (motion.compare("Think_finish") == 0) || (motion.compare("0") == 0) || (motion.compare("1") == 0) || (motion.compare("2") == 0) || (motion.compare("3") == 0) || (motion.compare("4") == 0)) {
    	for(int mm_iter = 0; mm_iter < time_arr.size(); mm_iter++) {
    	    while (curr_time - start_time < time_arr[mm_iter]) { curr_time = time(0); }
    	    moveArmWithSpeed(angle_input[mm_iter][7], angle_input[mm_iter][2], -angle_input[mm_iter][3], angle_input[mm_iter][6], angle_input[mm_iter][0], -angle_input[mm_iter][1], speed[mm_iter][7], speed[mm_iter][2], speed[mm_iter][3], speed[mm_iter][6], speed[mm_iter][0], speed[mm_iter][1]);
        }

        // Reset arm positions
        if ((motion.compare("Think_finish") == 0) || (motion.compare("wave") == 0) || (motion.compare("0") == 0) || (motion.compare("1") == 0) || (motion.compare("2") == 0) || (motion.compare("3") == 0))
        {
    	   moveArm(angle_input[angle_input.size() - 1][7], 0, 0, angle_input[angle_input.size() - 1][6], 0, 0, 80);
           sleep(1);
    	   moveArm(0, 0, 0, 0, 0, 0, 50);
        }
    }
    // else if ((motion.compare("good") == 0) || (motion.compare("2") == 0)) {
    //     for(int mm_iter = 0; mm_iter < time_arr.size(); mm_iter++) {
    //         while (curr_time - start_time < time_arr[mm_iter]) { curr_time = time(0);  }
    //         movePantilt(angle_input[mm_iter][5]*3, angle_input[mm_iter][4]*3, 0, 35, 0);
    //     }
    //     movePantilt(0, 0, 0, 35, 0);      // Reset arm positions
    // } else {
    //     for(int mm_iter = 0; mm_iter < time_arr.size(); mm_iter++) {
    //         while (curr_time - start_time < time_arr[mm_iter]) { curr_time = time(0);   }
    //         movePantilt(angle_input[mm_iter][4]*3, angle_input[mm_iter][5]*3, 0, 35, 0); 
    //     }
    //     movePantilt(0, 0, 0, 35, 0);      // Reset arm positions
    // }
    
    gesturePublisher.publish(msg_pub);
    recent_action = angle_input[angle_input.size() - 1];
    // cout << "\n";
    // for (int i = 0; i < angle_input.size(); i++) {
    //     cout << "{";
    //     for (int j = 0; j < angle_input[i].size(); j++) {
    //         cout << angle_input[i][j] << ", ";
    //     }
    // cout << "}\n";
    // }
}

time_t start = time(NULL);
time_t rec = time(NULL);

void faceResponse(const std_msgs::Int32::ConstPtr& msg)
{
    // ROS_INFO("Face Detected:");
    rec = time(NULL);
    if(face_detected == 0) {
        start = time(NULL);
        face_detected = 1;
        applyMotions("wave");
    }

    // If a face hasnt been detected in >20 seconds, wave at the next face it sees 
    if(face_detected == 1 && (rec - start >= 20)) {
        face_detected = 0;
    }
    start = time(NULL);
}


void gptResponse(const std_msgs::String::ConstPtr& msg)
{
    applyMotions(msg->data.c_str());
}

void shitCallback(const std_msgs::String::ConstPtr& msg)
{
    applyMotions(msg->data.c_str());
}

int main(int argc, char *argv[])
{
	cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n\n\n\n\n\n\n\n\n\nDDDDDDDDD";
    
    ros::init(argc, argv, "Silbot3_tele_keyboard");


	ros::NodeHandle node;
    armPublisher = node.advertise<silbot3_msgs::Device_Arm_Msg>("/DeviceNode/Arm/commands", 1000);
    pantiltPublisher = node.advertise<silbot3_msgs::Device_ErobotPantilt_Msg>("/DeviceNode/Pantilt/commands", 1000);
    gesturePublisher = node.advertise<std_msgs::Int32>("gesturetopic", 1000);
    gptSubscriber = node.subscribe("gpttopic", 1000, gptResponse);
    faceSubscriber = node.subscribe("Face_Detected_String", 100, faceResponse);
    shitSubscriber = node.subscribe("shittopic", 1000, shitCallback);
    initMotions(1);

    ros::spin();
}
