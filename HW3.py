#Alvaro Guerra
#9/20/24
#HW3

import numpy as np
import math

#1
#M = int(float(input("Select material system (1-5)")))
#angle = int(float(input("Input integer angle orientation")))
M = 2
angle = 45
global_stress_xx = -20000
global_stress_yy = 20000
global_shear = 0

#a
mat_type = {1:'T300/5208_graphite_epoxy', 2:'B(4)/5505_boron_epoxy', 3:'AS/3501_graphite_epoxy', 4:'Scotchply_1002_glass_epoxy', 5:'Kevlar49_aramid_epoxy'}
material_select = mat_type[M]


e11 = np.array([[26.25*1000000],
               [29.59*1000000],
               [20.01*10000000],
               [5.6*1000000],
               [11.02*1000000]
                    ])
e11_mat =  e11[M-1,0]     
               
e22 = np.array([[1.49*1000000],
               [2.68*1000000],
               [1.3*1000000],
               [1.2*1000000],
               [0.8*1000000]
                    ])

e22_mat =  e22[M-1,0] 

g12 = np.array([[1.04*1000000],
               [0.81*1000000],
               [1.03*1000000],
               [0.6*1000000],
               [0.33*1000000]
                    ])
g12_mat =  g12[M-1,0] 

v12 = np.array([[0.28],
               [0.23],
               [0.3],
               [0.26],
               [0.34]
                    ])
v12_mat =  v12[M-1,0] 

v21 = (v12_mat*e22_mat)/e11_mat

theta = angle*(2*math.pi)/360
c = math.cos(theta)
s = math.sin(theta)

#G-L stress transformation matrix
T = np.array([[c**2, s**2, 2*s*c],
                       [s**2, c**2, -2*s*c],
                       [-s*c, s*c, (c**2)-(s**2)]
                       ])

#G-L strain transformation matrix
T_epsilon = np.array([[c**2, s**2, s*c],
                       [s**2, c**2, -s*c],
                       [-2*s*c, 2*s*c, (c**2)-(s**2)]
                       ])
#Modulus Matrix in Local Coord (kpsi)
Q = np.array([[(e11_mat/(1-v12_mat*v21)), (v21*e11_mat)/(1-v12_mat*v21), 0],
              [((v12_mat*e22_mat)/(1-v12_mat*v21)), ((e22_mat)/(1-v12_mat*v21)), 0],
              [0, 0, g12_mat]
             ])
#Compliance Matrix in Ply/Local Coords (1/kpsi)
S = np.linalg.inv(Q)

#Modulus Matrix in Laminate/Global coords (kpsi)
Q_capybara = np.matmul((np.linalg.inv(T)),Q)
Q_bar = np.matmul(Q_capybara,T_epsilon)

#Global Stress & Strain Matrix (ksi)
global_stresses =  np.array([[global_stress_xx],
                                  [global_stress_yy],
                                  [global_shear],
                                  ])

global_strains = np.matmul(np.linalg.inv(Q_bar),global_stresses)

#Local/Ply Stress/Strains (kpsi)
local_stress = np.matmul(T,global_stresses)

local_strain = np.matmul(np.linalg.inv(Q), local_stress)

#Global Strain (Utilizing local->global transformation)
#global_strains = global_strains2
global_strain2 = np.matmul(np.linalg.inv(T_epsilon), local_strain)

print("\n")
print("Question 1 Results:", "\n")
print("a) Local stress:", "\n", local_stress)
print("\n")
print("b) Local strain:","\n",  local_strain)
print("\n")
print("c) Global strain:","\n",  global_strain2)
print("\n")

###2 unbalanced & symmetric

#a)
#[0,60,90,60]s
thickness = 0.005
ply_num = 7
ply_angle_dict = {1:0, 2:60, 3:90, 4:60, 5:60, 6:90, 7:60, 8:0}
ply_zdist = {1:-0.02,2:-0.015,3:-0.01, 4:0,5:0.005,6:0.01,7:0.015,8:0.02}
M = 3

e11_mat =  e11[M-1,0]     
e22_mat =  e22[M-1,0] 
g12_mat =  g12[M-1,0] 
v12_mat =  v12[M-1,0] 
v21 = (v12_mat*e22_mat)/e11_mat

t_lam = thickness * len(ply_angle_dict)
z0 = -t_lam/2

T_eps_list = []
T_list = []
Q_list = []
Q_bar_list = []
A_list = []
B_list = []
D_list = []

for j in ply_angle_dict:
    z_uppersurf = z0 + j*thickness
    z_mid = z0 - (thickness/2) + j*thickness
    angle = ((math.pi)/180)*ply_angle_dict[j]
    c = math.cos(angle)
    s = math.sin(angle)
    threshold = 1e-4
    if abs(c) < threshold:
        c = 0
    if abs(s) < threshold:
        s = 0
        
    T_list.append(np.array([[c**2, s**2, 2*s*c],
                           [s**2, c**2, -2*s*c],
                           [-s*c, s*c, (c**2)-(s**2)]
                           ]) )
    T_eps_list.append(np.array([[c**2, s**2, s*c],
                           [s**2, c**2, -s*c],
                           [-2*s*c, 2*s*c, (c**2)-(s**2)]
                           ]))
    Q_list.append(np.array([[(e11_mat/(1-v12_mat*v21)), (v21*e11_mat)/(1-v12_mat*v21), 0],
                  [((v12_mat*e22_mat)/(1-v12_mat*v21)), ((e22_mat)/(1-v12_mat*v21)), 0],
                  [0, 0, g12_mat]
                 ]))
    
    T_inv = np.linalg.inv(T_list[j-1])
    Q_bar_list.append(np.matmul(T_inv, np.matmul(Q_list[j-1], T_eps_list[j-1])))

    A_list.append(Q_bar*thickness)
    B_list.append(0.5*(Q_bar_list[j-1]*((z_mid**2)-(z_mid-thickness)**2)))
    D_list.append((1/3)*(Q_bar_list[j-1]*((z_mid**3)-(z_mid-thickness)**3)))

print(A_list[3])























###3 balance and unsymmetric


