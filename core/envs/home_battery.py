import os
import numpy as np
import matplotlib.pyplot as plt

class Home_Battery():
    def __init__(self, rho1 = 4.0, rho2 = 30.0):
        self.state_dim = 2
        self.num_actions = 4
        self.rho1 = abs(rho1)         # Amplification parameter for time of use prices
        self.rho2 = abs(rho2)         # Amplification parameter for time of use prices
        self.load_demand_profile()
        self.set_battery_specs()
        self.set_time_of_use_prices()
        self.reset()                   # Initial reset

        self.record_data = False

    def reset(self):
        self.time_slot = 0
        self.cumulative_energy_cost = 0
        self.battery_soc = np.clip(0.1*np.random.randint(0, 11)*self.battery_capacity, self.battery_min, self.battery_max)
        return self.observe()

    def observe(self):
        normalized_time = (self.time_slot - 0.5*self.num_time_slots)/(0.5*self.num_time_slots)
        normalized_soc = (self.battery_soc - 0.5*self.battery_max)/(0.5*self.battery_max)
        return np.array([normalized_time, normalized_soc])

    def step(self, action):
        action_in = int(action)
        assert action_in in [0, 1, 2, 3], 'ERROR: action must an integer within [0, 3].'
        energy_demand, energy_exchange = self.dynamics(action_in)
        reward = self.reward(energy_exchange)
        self.cumulative_energy_cost += energy_exchange * self.tou_prices[self.time_slot]

        if(self.record_data):
            self.data_log.append(np.array([self.time_slot, energy_demand, self.battery_soc, self.tou_prices[self.time_slot], action_in]))

        self.time_slot = (self.time_slot + 1) % self.num_time_slots
        return self.observe(), reward, False

    def dynamics(self, action):
        # Actions:  0-Buy, 1-Charge, 2-Discharge, 3-Sell
        energy_demand = self.get_energy_demand(self.time_slot)
        delta_energy_demand = (self.battery_soc - self.battery_min) - energy_demand
        demand_after_discharge = np.maximum(-delta_energy_demand, 0)
        energy_to_sell = self.energy_unit * (delta_energy_demand >= self.energy_unit).astype(float)
        energy_exchange = [energy_demand,
                           energy_demand + self.energy_unit,
                           demand_after_discharge,
                           demand_after_discharge - energy_to_sell]
        energy_charged = [0,
                          self.energy_unit,
                          -energy_demand,
                          -(energy_demand + energy_to_sell)]

        self.battery_soc = np.clip(self.battery_soc + energy_charged[action], self.battery_min, self.battery_max)
        return energy_demand, energy_exchange[action]

    def get_energy_demand(self, time_slot):
        base_load = self.base_load[time_slot]
        energy_demand = np.random.normal(base_load, scale=0.1)
        return energy_demand

    def reward(self, purchased_energy):
        return -self.rho1 * np.exp((self.tou_prices[self.time_slot] - self.max_tou_price) * self.rho2) * purchased_energy

    def get_available_actions(self):
        # Actions:  0-Buy, 1-Charge, 2-Discharge, 3-Sell
        available_actions = [0,1,2,3]
        if(self.battery_soc <= self.battery_min):
            available_actions = [0, 1]
        elif(self.battery_soc <= (self.battery_min + self.energy_unit)):
            available_actions = [0,1,2]
        elif(self.battery_soc > (self.battery_max - self.energy_unit)):
            available_actions = [0,2,3]
        return available_actions

    def load_demand_profile(self):
        directory = os.path.dirname(__file__)
        data_path = os.path.join(directory, 'energy_demands/e1_dynamic_2001.xls')
        base_load_profile = np.loadtxt(data_path, dtype='str', delimiter='\t', skiprows=1)
        self.base_load = base_load_profile[-1, 2:].astype(float).reshape(-1)
        self.num_time_slots = self.base_load.shape[0]

    def show_demand_profile(self):
        plt.figure(figsize = (10, 5))
        plt.bar(np.arange(self.num_time_slots), self.base_load, color='tab:orange', alpha=0.5)
        plt.ylabel('Demand (kWh)')
        plt.xlabel('Time (h)')
        plt.title('Residential demand profile within a ' + str(self.num_time_slots) + 'h window')
        ticks = np.arange(self.num_time_slots).astype(int)
        plt.xticks(np.arange(self.num_time_slots), ticks)
        plt.show()

    def set_battery_specs(self):
        self.battery_capacity = 30                      # kWh
        self.battery_min = 0.02*self.battery_capacity   # kWh
        self.battery_max = 0.98*self.battery_capacity   # kWh
        self.energy_unit = self.battery_capacity/10     # kWh

    def set_time_of_use_prices(self):
        self.tou_prices = np.array([.219]*9 + [.246]*3 + [.27]*6 + [.246]*4 + [.219]*2)
        self.max_tou_price = np.max(self.tou_prices)

    def enable_data_recorder(self):
        self.record_data = True
        self.data_log = []

    def show_recorded_data(self, num_time_slots = 3*24, figsize=(12, 6)):
        if(self.record_data):
            data = np.stack(self.data_log)
            total_time_slots = np.arange(data.shape[0])

            plt.figure(figsize=figsize)
            plt.subplot(211)
            plt.step(total_time_slots, data[:, 1]/np.max(data[:, 1]), label='Demand', linewidth=2)
            plt.step(total_time_slots, data[:, 2]/np.max(data[:, 2]), label='Battery SOC', linewidth=2)
            plt.step(total_time_slots, data[:, 3]/np.max(data[:, 3]), label='TOU price', linewidth=2)
            plt.ylabel('Normalized value')
            plt.legend(loc=4)
            plt.grid()
            plt.xlim((0, num_time_slots))
            plt.title('Total energy cost ($): ' + str(np.round(self.cumulative_energy_cost, 2)))

            plt.subplot(212)
            plt.step(total_time_slots, data[:, 4], label='Action taken', color='tab:red', linewidth=2)
            plt.xlabel('Time [h]')
            plt.yticks(np.arange(4),('Buy', 'Charge', 'Discharge', 'Sell'))
            plt.legend(loc=4)
            plt.grid()
            plt.xlim((0, num_time_slots))
            plt.show()

    def get_time_of_use_prices(self):
        return self.tou_prices

    def get_cumulative_energy_cost(self):
        return self.cumulative_energy_cost
