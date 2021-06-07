import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class range_image_node():

    def __init__(self):
        rospy.init_node('listener', anonymous=True)
        self.r = rospy.Rate(10)
        self.raw_image_pub = rospy.Publisher('range_image_raw', Float32MultiArray, queue_size=10)
        rospy.Subscriber("/model_name", String, self.model_name_callback)
        self.r.sleep()
        rospy.Subscriber("/" + self.model_name + "/range_finder/range_image", Image, self.range_callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

    def range_callback(self, data):
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data)
        arr = Float32MultiArray()
        arr.data = list(cv_image.reshape(4096))
        #print(arr)
        #print("image:", cv_image)
        self.raw_image_pub.publish(arr)


    def model_name_callback(self, data):
        self.model_name = data.data


node_class = range_image_node()
