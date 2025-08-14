import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import asyncio
from torch.optim.lr_scheduler import StepLR

class ActorCritic(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # with torch.autocast(device_type='cuda', dtype=torch.float16):  # 混合精度
        shared = self.shared(x)
        action_logits = self.actor(shared)

        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(shared)
        state_value=state_value.squeeze()
        return action_probs, state_value

class SPGG(nn.Module):
    def __init__(self, L_num, device, alpha, gamma, clip_epsilon, r, epochs, 
                now_time, question, ppo_epochs, batch_size, gae_lambda, 
                output_path, delta, rho, 
                beta, tau, zeta):
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epochs = epochs
        self.question = question
        self.now_time = now_time
        
        # PPO超参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.delta=delta # w_cl
        self.rho=rho # w_ent

        self.output_path=output_path
        
        # 神经网络
        self.policy = ActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.5)  # 每1000步学习率降低10%
        
        # 邻域卷积核
        self.neibor_kernel = torch.tensor(
            [[[[0,1,0], [1,1,1], [0,1,0]]]], 
            dtype=torch.float32, device=device
        )
        
        # 初始化状态
        self.initial_state = self._init_state(question)
        self.current_state = self.initial_state.clone()
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        # 团队效用参数
        self.team_size = 5  # 默认团队大小(4邻居+自己)

        # 新增以下参数
        self.beta = beta  # 自适应权重敏感度
        self.tau = tau   # 团队效用阈值
        self.eta = nn.Parameter(torch.tensor(0.1))  # 可学习的拉格朗日乘子
        self.zeta = zeta  # 对偶学习率
        self.hist_ind_rewards = []  # 新增历史记录缓冲区
        self.hist_team_rewards = []

    def _init_state(self, question):
        if question == 1: # question 1: 伯努利分布随机50%概率背叛和合作
            state = torch.bernoulli(torch.full((self.L_num, self.L_num), 0.5))
        elif question == 2: # question 2: 上半背叛，下半合作
            state = torch.zeros(self.L_num, self.L_num)
            state[self.L_num//2:, :] = 1
        elif question == 3:  # question 3: 全背叛
            state = torch.zeros(self.L_num, self.L_num)
        elif question == 4:
            # 创建一个 L x L 的零矩阵
            state = torch.zeros((self.L_num, self.L_num))
            # 填充交替的 0 和 1
            for i in range(self.L_num):
                for j in range(self.L_num):
                    if (i + j) % 2 == 0:
                        state[i, j] = 1
        return state.to(self.device)

    def encode_state(self, state_matrix):
        """将 2D 网格转换为 4D 张量后填充"""
        # 添加 batch 和 channel 维度 [B, C, H, W]
        state_4d = state_matrix.float().unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        # 使用正确的填充参数格式 (padding_left, padding_right, padding_top, padding_bottom)
        padded = F.pad(state_4d, (1, 1, 1, 1), mode='circular')  # 四周各填充1
        
        # 计算邻域合作数
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()  # [L, L]
        global_coop = torch.mean(state_matrix.float())
        return torch.stack([
            state_matrix.float().squeeze(),
            neighbor_coop,
            global_coop.expand_as(state_matrix)
        ], dim=-1).view(-1, 3)
    
    def calculate_adaptive_weight(self):
        if len(self.hist_ind_rewards) == 0 or len(self.hist_team_rewards) == 0:
            return 0.5  # 默认初始权重
        
        sum_ind = torch.sum(torch.stack(self.hist_ind_rewards))
        sum_team = torch.sum(torch.stack(self.hist_team_rewards))
        ratio = sum_team / (sum_ind + 1e-8)
        return torch.sigmoid(self.beta * ratio)
    def calculate_reward(self, state_matrix):
        """修改后的奖励计算函数"""
        # 1. 原有个人收益计算（保持不变）
        padded = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()
        
        c_single_profit = (self.r * neighbor_coop / 5) - 1
        d_single_profit = (self.r * neighbor_coop / 5)
        
        padded_c_profit = F.pad(c_single_profit.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        padded_d_profit = F.pad(d_single_profit.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        c_total_profit = F.conv2d(padded_c_profit, self.neibor_kernel).squeeze() + c_single_profit
        d_total_profit = F.conv2d(padded_d_profit, self.neibor_kernel).squeeze() + d_single_profit
        personal_reward = torch.where(state_matrix.bool(), c_total_profit, d_total_profit)
        
        # 2. 计算团队效用（保持不变）
        team_utility = self.calculate_team_utility(state_matrix)
        
        # 3. 记录历史数据（新增部分）
        self.hist_ind_rewards.append(personal_reward.mean().detach())
        self.hist_team_rewards.append(team_utility.mean().detach())
        
        # 4. 计算自适应权重（新增部分）
        if len(self.hist_ind_rewards) > 0 and len(self.hist_team_rewards) > 0:
            # sum_ind = torch.sum(torch.stack(self.hist_ind_rewards[-100:]))  # 最近100步
            # sum_team = torch.sum(torch.stack(self.hist_team_rewards[-100:]))
            sum_ind = torch.sum(torch.stack(self.hist_ind_rewards))  # 最近100步
            sum_team = torch.sum(torch.stack(self.hist_team_rewards))
            ratio = sum_team / (sum_ind + 1e-8)
            w_t = torch.sigmoid(self.beta * ratio)
        else:
            w_t = 0.5  # 默认初始权重
        
        # 5. 组合奖励（修改部分）
        combined_reward = (1 - w_t) * personal_reward + w_t * team_utility
        
        return combined_reward, personal_reward, team_utility  # 现在返回三个值

    def _make_batch(self, states, actions, old_log_probs, advantages, returns):
        perm = torch.randperm(len(states))
        for i in range(0, len(states), self.batch_size):
            idx = perm[i:i+self.batch_size]
            yield (states[idx], actions[idx], old_log_probs[idx], advantages[idx], returns[idx])

    def run(self):
        coop_rates = []
        defect_rates = []
        total_values = []

        # 新增缓冲区（在epoch循环外部）
        self.personal_rewards = []  # 新增个人奖励缓冲区
        self.team_utilities = []   # 新增团队效用缓冲区
        
        for epoch in tqdm(range(self.epochs)):
            self.epoch=epoch
            action, log_prob = self.choose_action(self.current_state)
            next_state=action

            # 修改奖励获取方式（原self.calculate_reward(next_state)替换为）
            reward, personal_reward, team_utility = self.calculate_reward(next_state)
            done = torch.zeros_like(next_state, dtype=torch.bool)
            
            # 存储transition（修改部分）
            self.states.append(self.encode_state(self.current_state).view(self.L_num,self.L_num,3))
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)                   # 组合奖励
            self.personal_rewards.append(personal_reward) # 新增存储
            self.team_utilities.append(team_utility)      # 新增存储
            self.next_states.append(self.encode_state(next_state).view(self.L_num,self.L_num,3))
            self.dones.append(done)
            
            if epoch==0:
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic(self.current_state, epoch, self.r, profit_matrix))
                coop_rate = self.current_state.float().mean().item()
                defect_rate = 1 - coop_rate
                total_value = reward.sum().item()
                
                coop_rates.append(coop_rate)
                defect_rates.append(defect_rate)
                total_values.append(total_value)

            # 修改PPO更新条件判断（原if len(self.states) >= self.batch_size * self.ppo_epochs:）
            if len(self.states) >= self.batch_size * self.ppo_epochs:
                self.constrained_optimization_step()  # 替换原有的ppo_update()
                self.current_state = next_state
                self._reset_buffer()  # 清空所有缓冲区
            else:
                self.current_state = next_state       

            # self.current_state = next_state
            # 在关键时间点保存快照（与原Q-learning相同）
            if (epoch+1 in [1, 10, 100, 1000, 10000, 100000]):
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic(self.current_state, epoch+1, self.r, profit_matrix))

            # 收集数据
            coop_rate = self.current_state.float().mean().item()
            defect_rate = 1 - coop_rate
            total_value = reward.sum().item()
            
            coop_rates.append(coop_rate)
            defect_rates.append(defect_rate)
            total_values.append(total_value)
        
        return defect_rates, coop_rates, [], [], total_values

    def save_data(self, data_type, name, r, data):
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        # output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}.txt', data)

    # def mkdir(self, path):
    #     os.makedirs(path, exist_ok=True)
    
    async def shot_pic(self, type_t_matrix, epoch, r, profit_data):
        """保存策略矩阵快照与数据文件（与原Q-learning代码相同格式）"""
        plt.clf()
        plt.close("all")
        
        # 创建输出目录
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        os.makedirs(profit_dir, exist_ok=True)

        # =============================================
        # 1. 保存策略矩阵图
        # =============================================
        # 生成策略矩阵可视化
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        ax1.axis('off')
        # 设置 figure 的边框为黑色
        fig1.patch.set_edgecolor('black')
        fig1.patch.set_linewidth(2)  # 设置边框线宽
        
        # 创建颜色映射（黑:背叛者，白:合作者）
        color_map = {
            0: [0, 0, 0],    # 黑色
            1: [1, 1, 1]     # 白色
        }
        
        # 转换为RGB图像
        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color
        
        # 绘图设置
        ax1.imshow(strategy_image, interpolation='none')
        ax1.axis('off')
        for spine in ax1.spines.values():
            spine.set_linewidth(3)
            
        # 保存图片
        # plt.savefig(f'{img_dir}/t={epoch}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        fig1.savefig(f'{img_dir}/t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)
        
        # =============================================
        # 2. 保存收益热图
        # =============================================
        # 2. 处理profit数据
        if isinstance(profit_data, tuple):
            # 如果是元组，提取combined_reward和team_utility
            combined_reward, _, team_utility = profit_data
            profit_matrix = combined_reward
        else:
            profit_matrix = profit_data
        
        # 确保是张量并转移到CPU
        if not isinstance(profit_matrix, torch.Tensor):
            profit_matrix = torch.tensor(profit_matrix, device=self.device)
        profit_matrix = profit_matrix.cpu().numpy()

        fig2 = plt.figure(figsize=(8, 8))
        ax2 = fig2.add_subplot(1, 1, 1)

        # 绘制热图
        vmin, vmax = 0, 9
        # profit_data = profit_matrix.cpu().numpy()
        im = ax2.imshow(profit_matrix, vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')
        
        # 添加颜色条
        
        cbar2 = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=28)
        cbar2.set_ticks(np.arange(0, 9, 1))
        # cbar.set_label('Profit Value')
        
        # 设置坐标轴
        ax2.set_xticks(np.arange(0, self.L_num, max(1, self.L_num//5)))
        ax2.set_yticks(np.arange(0, self.L_num, max(1, self.L_num//5)))
        ax2.grid(False)

        ax2.axis('off')
        # 设置 figure 的边框为黑色
        # fig2.patch.set_edgecolor('black')
        # fig2.patch.set_linewidth(2)  # 设置边框线宽
        
        # 保存收益热图
        fig2.savefig(f'{img_dir}/profit_t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)

        # 3. 如果是元组输入，额外绘制团队效用
        if isinstance(profit_data, tuple):
            _, _, team_utility = profit_data
            if not isinstance(team_utility, torch.Tensor):
                team_utility = torch.tensor(team_utility, device=self.device)
            team_utility = team_utility.cpu().numpy()
            
            fig3 = plt.figure(figsize=(8, 8))
            ax3 = fig3.add_subplot(1, 1, 1)

            ax3.axis('off')
            # 设置 figure 的边框为黑色
            # fig3.patch.set_edgecolor('black')
            # fig3.patch.set_linewidth(2)  # 设置边框线宽

            im3 = ax3.imshow(team_utility, vmin=vmin, vmax=vmax, cmap='plasma', interpolation='none')
            cbar3 = fig3.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.ax.tick_params(labelsize=28)
            cbar3.set_ticks(np.arange(0, 9, 1))
            # cbar3.set_label('Team Utility')
            fig3.savefig(f'{img_dir}/team_utility_t={epoch}.pdf', 
                        format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig3)

        # =============================================
        # 3. 保存矩阵数据文件
        # =============================================
        np.savetxt(f'{matrix_dir}/T{epoch}.txt',
                    type_t_matrix.cpu().numpy(), fmt='%d')
        np.savetxt(f'{profit_dir}/T{epoch}.txt',
                    profit_matrix, fmt='%.4f')
        np.savetxt(f'{profit_dir}/T{epoch}_team_utility.txt',
                    team_utility, fmt='%.4f')
        return 0

    def _reset_buffer(self):
        """显式释放所有缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.personal_rewards = []
        self.team_utilities = []
        torch.cuda.empty_cache()
    
    # 修改经验存储逻辑
    def _store_transition(self, state, action, log_prob, reward, next_state, done):
        """存储时分离梯度"""
        self.states.append(state.detach().cpu())  # 转移到CPU
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.next_states.append(next_state.detach().cpu())
        self.dones.append(done.detach().cpu())

    def choose_action(self, state_matrix):
        with torch.no_grad():
            features = self.encode_state(state_matrix)
            probs, _ = self.policy(features)
            dist = Categorical(probs)
            actions = dist.sample()
        return actions.view_as(state_matrix), dist.log_prob(actions).view_as(state_matrix)
    
    def calculate_team_utility(self, state_matrix):
        """计算每个位置的团队效用"""
        # 1. 对状态矩阵进行padding处理（环形边界）
        padded = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        # 2. 计算每个位置的邻域合作者数量（4邻居）
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()
        
        # 3. 计算团队合作者数量 (n_C + 1 if self is cooperator)
        team_coop = neighbor_coop + state_matrix.float()
        
        # 4. 计算团队效用 (r * team_coop / team_size - 1 if cooperate else 0)
        team_utility = torch.where(
            state_matrix.bool(),
            (self.r * team_coop / self.team_size) - 1,  # 合作者的团队效用
            torch.zeros_like(team_coop)  # 背叛者不贡献团队效用
        )
        
        return team_utility
    
    def _reset_histories(self):
        """专门清空历史记录缓冲区"""
        del self.hist_ind_rewards[:]
        del self.hist_team_rewards[:]
        torch.cuda.empty_cache()

    # 新增约束优化方法（在SPGG类中添加）
    def constrained_optimization_step(self):
        """整合约束优化的PPO更新"""
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.stack(self.dones)
        
        # 计算平均团队效用（新增）
        avg_team_utility = torch.mean(torch.stack(self.team_utilities))
        constraint_violation = torch.relu(self.tau - avg_team_utility)
        
        # 原有advantage计算
        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)
        
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            dones_float = dones[t].float()
            psi = rewards[t] + self.gamma * next_values[t] * (1 - dones_float) - values[t]
            advantages[t] = psi + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values
        
        # PPO更新循环
        for _ in range(self.ppo_epochs):
            for batch in self._make_batch(states, actions, old_log_probs, advantages, returns):
                state_b, action_b, old_log_b, adv_b, ret_b = batch
                probs, value_pred = self.policy(state_b)
                dist = Categorical(probs)
                
                # 计算各项损失
                log_probs = dist.log_prob(action_b).view_as(action_b)
                ratio = (log_probs - old_log_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                if value_pred.dim() == 2:  # 如果是 [L_num, L_num]
                    value_pred = value_pred.view(ret_b.shape[0],value_pred.shape[0],value_pred.shape[1])  # 变为 [b, L_num, L_num]
                critic_loss = F.mse_loss(value_pred, ret_b)
                entropy = dist.entropy().mean()
                
                # 修改后的损失函数（新增约束项）
                loss = (actor_loss + 
                    self.delta * critic_loss - 
                    self.rho * entropy + 
                    self.eta * constraint_violation)
                
                # 梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        # 更新拉格朗日乘子（新增）
        self.eta.data += self.zeta * constraint_violation.item()
        
        # 清空历史记录（新增）
        self._reset_histories()