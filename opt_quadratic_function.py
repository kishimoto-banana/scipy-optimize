import scipy.optimize

def objective_function(theta):
    '''目的関数''' 
    return (theta - 2) ** 2

def gradient(theta):
    '''勾配'''
    return 2 * (theta - 2)

def main():

    # 勾配有り
    theta_opt = scipy.optimize.fmin_bfgs(f=objective_function, x0=[0], fprime=gradient)
    print(f'勾配有り\nθ={theta_opt}')

    # 勾配有り
    theta_opt = scipy.optimize.fmin_bfgs(f=objective_function, x0=[0])
    print(f'勾配無し\nθ={theta_opt}')


if __name__ == '__main__':
    main()
