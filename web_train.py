import gymnasium as gym
import numpy as np
import os
import threading
import time
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from agent import DQNAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cartpole-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

training_state = {
    'running': False,
    'paused': False,
    'episode': 0,
    'total_episodes': 500,
    'epsilon': 1.0,
    'rewards': [],
    'avg_rewards': [],
    'losses': [],
    'cart_state': None,
    'agent': None,
    'env': None
}


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to server'})
    emit('training_state', {
        'running': training_state['running'],
        'paused': training_state['paused'],
        'episode': training_state['episode'],
        'total_episodes': training_state['total_episodes'],
        'rewards': training_state['rewards'],
        'avg_rewards': training_state['avg_rewards']
    })


@socketio.on('start_training')
def handle_start_training():
    if not training_state['running']:
        training_state['running'] = True
        training_state['paused'] = False
        training_state['episode'] = 0
        training_state['rewards'] = []
        training_state['avg_rewards'] = []
        training_state['losses'] = []
        socketio.start_background_task(train_loop)
        emit('status', {'message': 'Training started'}, broadcast=True)


@socketio.on('pause_training')
def handle_pause_training():
    training_state['paused'] = not training_state['paused']
    status = 'paused' if training_state['paused'] else 'resumed'
    emit('status', {'message': f'Training {status}'}, broadcast=True)


@socketio.on('stop_training')
def handle_stop_training():
    training_state['running'] = False
    emit('status', {'message': 'Training stopped'}, broadcast=True)


def train_loop():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    training_state['agent'] = agent
    training_state['env'] = env
    
    num_episodes = training_state['total_episodes']
    max_steps = 500
    
    socketio.emit('status', {'message': f'Training DQN on CartPole-v1, Device: {agent.device}'})
    
    for episode in range(num_episodes):
        if not training_state['running']:
            break
            
        while training_state['paused']:
            socketio.sleep(0.1)
            if not training_state['running']:
                break
        
        if not training_state['running']:
            break
            
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            if not training_state['running']:
                break
                
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                episode_loss.append(loss)
            
            # Emit cart state for animation
            socketio.emit('cart_state', {
                'position': float(next_state[0]),
                'angle': float(next_state[2]),
                'step': step
            })
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        agent.decay_epsilon()
        training_state['episode'] = episode + 1
        training_state['epsilon'] = agent.epsilon
        training_state['rewards'].append(episode_reward)
        
        avg_reward = np.mean(training_state['rewards'][-100:])
        training_state['avg_rewards'].append(avg_reward)
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        training_state['losses'].append(avg_loss)
        
        # Emit episode update
        socketio.emit('episode_update', {
            'episode': episode + 1,
            'total_episodes': num_episodes,
            'reward': float(episode_reward),
            'avg_reward': float(avg_reward),
            'epsilon': float(agent.epsilon),
            'loss': float(avg_loss)
        })
        
        if avg_reward >= 475 and episode >= 100:
            socketio.emit('status', {'message': f'Solved in {episode + 1} episodes!'})
            break
        
        socketio.sleep(0.01)  # Small delay for visualization
    
    env.close()
    training_state['running'] = False
    
    # Save model
    os.makedirs('models', exist_ok=True)
    agent.save('models/dqn_cartpole.pth')
    socketio.emit('training_complete', {
        'message': 'Training complete! Model saved.',
        'final_avg_reward': float(training_state['avg_rewards'][-1]) if training_state['avg_rewards'] else 0
    })


if __name__ == '__main__':
    print("Starting CartPole Training Server...")
    print("Open http://localhost:5001 in your browser")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
