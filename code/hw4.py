#-------------------------------Importation of required libraries-------------------------------------
from sympy import *
import matplotlib.pyplot as plt
import numpy as np
import math

#------------------------------Declaration of variables and symbols-----------------------------------

theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7  = symbols('th1, th2, th3, th4, th5, th6, th7', real=True)
theta = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7]
theta[2] = 0



#--------------------------Giving the values of d, a, alpha instead of symbols------------------------------------
d = [0]*7
d[0] = 0.3330
d[1] = 0
d[2] = 0.3160
d[3] = 0
d[4] = 0.3840
d[5] = 0
d[6] = -0.1070 - 0.1

a = [0]*7
a[0] = 0
a[1] = 0
a[2] = 0.0880
a[3] = -0.0880
a[4] = 0
a[5] = 0.0880
a[6] = 0

alpha = [0]*7
alpha[0] = pi/2
alpha[1] = -pi/2
alpha[2] = -pi/2
alpha[3] = pi/2
alpha[4] = pi/2
alpha[5] = -pi/2
alpha[6] = 0



#-----------------------------Calculations of Homogeneous Transformation Matrices---------------------------------
T_calc = []
for i in range(7):
    x = Matrix([[cos(theta[i]), -sin(theta[i])*cos(alpha[i]), sin(theta[i])*sin(alpha[i]), a[i]*cos(theta[i])],
         [sin(theta[i]), cos(theta[i])*cos(alpha[i]), -cos(theta[i])*sin(alpha[i]), a[i]*sin(theta[i])],
         [0, sin(alpha[i]), cos(alpha[i]), d[i]],
         [0, 0, 0, 1]
        ])
    T_calc.append(x)

final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]*T_calc[4]*T_calc[5]*T_calc[6]
final_trans = final_trans.evalf()

#---------------------------------T01--------------------------------------
final_trans = T_calc[0]
final_trans = final_trans.evalf()
z1 = final_trans[0:3,2]

#---------------------------------T02--------------------------------------
final_trans = T_calc[0]*T_calc[1]
final_trans = final_trans.evalf()
z2 = final_trans[0:3,2]

#---------------------------------T04--------------------------------------
final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]
final_trans = final_trans.evalf()
z4 = final_trans[0:3,2]

#---------------------------------T05--------------------------------------
final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]*T_calc[4]
final_trans = final_trans.evalf()
z5 = final_trans[0:3,2]

#---------------------------------T06--------------------------------------
final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]*T_calc[4]*T_calc[5]
final_trans = final_trans.evalf()
z6 = final_trans[0:3,2]

#---------------------------------T07--------------------------------------
final_trans = T_calc[0]*T_calc[1]*T_calc[2]*T_calc[3]*T_calc[4]*T_calc[5]*T_calc[6]
final_trans = final_trans.evalf()
z7 = final_trans[0:3,2]
xp = final_trans[0:3, 3]


#-----------------------------------Upper half of jacobian------------------------------
p1 = diff(xp, theta[0])
p2 = diff(xp, theta[1])
p4 = diff(xp, theta[3])
p5 = diff(xp, theta[4])
p6 = diff(xp, theta[5])
p7 = diff(xp, theta[6])

#--------------------------------------Jacobian Matrix----------------------------------
jac = Matrix([[p1[0], p2[0], p4[0], p5[0], p6[0], p7[0]], [p1[1], p2[1], p4[1], p5[1], p6[1], p7[1]], [p1[2], p2[2], p4[2], p5[2], p6[2], p7[2]], 
                [z1[0], z2[0], z4[0], z5[0], z6[0], z7[0]], [z1[1], z2[1], z4[1], z5[1], z6[1], z7[1]], [z1[2], z2[2], z4[2], z5[2], z6[2], z7[2]]])
print("\n\n==========================Jacobian=========================\n\n")
pprint(jac)

#--------------------------Iteration of Jacobian and Velocity Matrix--------------------
w = 2*pi/5
theta_inst = [0, 0, pi/2, 0, pi, 0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.axes.set_xlim(left=-10, right=90) 
ax.axes.set_ylim(bottom=-30, top=55) 
ax.axes.set_zlim(bottom=0, top=85) 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for t in range(100):
    p = jac.subs(theta_1, theta_inst[0]).subs(theta_2, theta_inst[1]).subs(theta_4, theta_inst[2]).subs(theta_5, theta_inst[3]).subs(theta_6, theta_inst[4]).subs(theta_7, theta_inst[5])
    jac_inv = p.inv()
    x_dot = Matrix([[0], [0.1*w/20*math.sin(w*t/20+pi/2)], [0.1*w/20*math.cos(w*t/20+pi/2)], [0], [0], [0]])
    q_dot = ((jac_inv*x_dot)% (2*pi)).evalf()  

    for i in range(len(q_dot)):
        theta_inst[i] += q_dot[i]

    inst_trans = final_trans.subs(theta_1, theta_inst[0]).subs(theta_2, theta_inst[1]).subs(theta_3, 0).subs(theta_4, theta_inst[2]).subs(theta_5, theta_inst[3]).subs(theta_6, theta_inst[4]).subs(theta_7, theta_inst[5])

    ax.scatter(inst_trans[0,3]*100, inst_trans[1,3]*100, inst_trans[2,3]*100)

plt.show()