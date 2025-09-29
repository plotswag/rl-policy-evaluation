# POLICY EVALUATION
## AIM
To simulate the Frozen-lake MDP and compare different policy functions.

## PROBLEM STATEMENT
The problem involves simulating a Frozen-lake MDP and defining various policy functions for it, these policy functions are later evaluated by a policy_evaluation() function which compares the value function of the policies passed as parameter. This is an experiment in reinforcement learning where you test different policies in FrozenLake, both by simulation (probability of reaching the goal) and by formal policy evaluation (computing expected long-term rewards).
## POLICY EVALUATION FUNCTION

<img width="685" height="130" alt="image" src="https://github.com/user-attachments/assets/834db01d-47b9-40d8-895e-5b7fc488ed1d" />

```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_v = np.zeros(len(P))  
    while True:
        v = np.zeros(len(P))  
        for s in range(len(P)):
            a = pi(s)  
            for prob, next_s, reward, done in P[s][a]:
                v[s] += prob * (reward + gamma * prev_v[next_s] * (not done))
        if np.max(np.abs(prev_v - v)) < theta:
            return v

        prev_v = v.copy()
    return v
```

## OUTPUT:
<img width="1805" height="605" alt="image" src="https://github.com/user-attachments/assets/6f8e9088-62e5-41c2-9ac0-968165596e7e" />

<img width="1807" height="184" alt="Screenshot 2025-09-29 085130" src="https://github.com/user-attachments/assets/a72eb392-71c2-4b06-85e8-f51e2b1a235b" />
<img width="1793" height="357" alt="image" src="https://github.com/user-attachments/assets/ccef8f3b-9cec-4618-8284-b1906cc9ef54" />

<img width="1807" height="273" alt="image" src="https://github.com/user-attachments/assets/992b6977-a3c6-48ab-aca3-a57717130402" />


## RESULT:

Thus we have successfully evaluated two different policies for a given env and compared their values functions.

