#!/usr/bin/env python
# coding: utf-8

# # Example3 - Divergence, Overflow And Python Tuples
# # $$h(x) = x^5 - 2x^4 + 2$$

# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# In[69]:


x_3 = np.linspace(-2.5, 2.5, 2000)
def h(x):
    return x**4 - 2*x**4 + 2
def dh(x):
    return 5*x**4 - 8*x**3


# In[70]:


def gradient_descent(derivative_func, initial_guess, multiplier=0.02,precision=0.01, max_iter=600):
    new_x = initial_guess
    x_list = [new_x]
    slope_list = [derivative_func(new_x)]

    for n in range(max_iter):
        previous_x = new_x
        gradient = derivative_func(previous_x)
        new_x = previous_x - multiplier * gradient
        step_size = abs(new_x - previous_x)
        x_list.append(new_x)
        slope_list.append(derivative_func(new_x))
        if step_size < precision:
            break
    return new_x,x_list, slope_list


# In[71]:


local_min,  list_x, deriv_list = gradient_descent(derivative_func=dh, initial_guess=-0.2,max_iter=10000)

plt.figure(figsize=[16,6])
plt.subplot(1,2,1)
plt.xlim(-1.2, 2.5)
plt.ylim(-1, 4)

plt.title('Cost Function', fontsize=17)
plt.xlabel('x', fontsize=16)
plt.ylabel('h(x)', fontsize=16)
plt.plot(x_3,h(x_3), color='#00FF00', linewidth=6,alpha=0.6)

plt.scatter(list_x,h(np.array(list_x)),color='red', s=100, alpha=0.6)

plt.grid()
plt.style.use('dark_background')

plt.subplot(1,2,2)

plt.title('Investition Budget', fontsize=17)
plt.xlabel('x',fontsize=16)
plt.ylabel('dh(x)', fontsize=16)
plt.xlim(-1 ,2)
plt.ylim(-4, 5)
plt.grid()
plt.plot(x_3,dh(x_3),color='#00FF00', linewidth=5, alpha=0.6)

plt.scatter(list_x,deriv_list,color='red', s=100, alpha=0.6)
plt.style.use('dark_background')
plt.show()


# In[72]:


import sys
from mpl_toolkits.mplot3d.axes3d import Axes3D
#help(sys)
sys.version
type(h(local_min))
sys.float_info.max


# # Example 4 - View Data Charts In  3D
# ## Minimise $$f(x, y) = \frac{1}{3^{-x^2 - y^2} + 1}$$
# Minimise $$f(x, y) = \frac{1} {r +1}$$ where $r$ is $3{-x^2 - y^2}$

# In[73]:


def f(x, y):
    r = 3**(-x**2 - y**2)
    return 1 / (r + 1)


# In[74]:


x_4 = np.linspace(start=-2, stop=2, num=200)
y_4 = np.linspace(start=-2, stop=2, num=200)

x_4, y_4 = np.meshgrid(x_4, y_4)
print('shape of X array', x_4.shape)


# In[79]:


fig = plt.figure(figsize=[16,13])
ax = fig.gca(projection='3d')
ax.set_xlabel('X', fontsize=20, color='red')
ax.set_ylabel('Y', fontsize=20, color='red')
ax.set_zlabel('f(x, y) - Cost', fontsize=20, color='red')
ax.plot_surface(x_4,y_4,f(x_4,y_4),cmap=cm.hsv, alpha=0.6)

plt.show()


# In[ ]:




