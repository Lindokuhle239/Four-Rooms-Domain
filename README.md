# Reinforcement Learning - Four Rooms Domain

## Assignment Overview
This project involves training a Q-learning agent to collect packages in a grid-world environment across three scenarios:
1. **Simple**: Collect 1 package
2. **Multi**: Collect 3 packages in any order
3. **RGB**: Collect 3 colored packages in strict Red→Green→Blue order

## Files
1. **scenariox.py**  
   Main implementation containing:  
   - QLearningAgent class with Q-learning algorithm  
   - Training logic for all scenarios  
   - Visualization functions  
   Usage: `python3 scenariox.py --scenario [simple|multi|rgb] (--stochastic)`

2. **FourRooms.py** (Provided)  
   Environment implementation - DO NOT MODIFY

3. **exploration_comparison.jpg**  
   Plot comparing ε-decay vs ε-fixed exploration strategies

4. **final_path_[scenario].jpg**  
   Agent path visualizations:  
   - simple: Single package collection  
   - multi: Multi-package collection  
   - rgb: Ordered package collection  

## How to Run
```bash
# Install dependencies
pip install numpy matplotlib

# Generate exploration strategy comparison plot (Scenario 1 analysis)
# 1. In scenariox.py, modify the bottom section to:
#    if __name__ == "__main__":
#        compare_exploration_strategies()
#        #main()
# 2. Run:
python3 scenariox.py

# Run scenarios (restore main() first):
# 1. In scenariox.py, ensure:
#    if __name__ == "__main__":
#        #compare_exploration_strategies()
#        main()
# 2. Execute:
python3 scenariox.py --scenario simple    # Basic package collection
python3 scenariox.py --scenario multi     # Multi-package collection  
python3 scenariox.py --scenario rgb       # Ordered collection
python3 scenariox.py --scenario simple --stochastic  # 20% action noise version
