import playground

bmi_update_rule = 'moving_average'
posterior_threshold = 0.01

if __name__ == "__main__":
    playground.run(bmi_update_rule, posterior_threshold, bmi_mode=False)
