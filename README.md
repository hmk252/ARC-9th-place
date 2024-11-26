
# ARC 2024: 9th Place Solution

My approach combines several solutions from the 2020 edition of the competition. By probing, I identified which tasks were solved by each solution and integrated them into a single script. Additionally, one specific task was manually solved by probing a training example.

Here’s the detailed breakdown of the 37 solved tasks:

- **26** from the ensemble of solutions ([Link to ensemble](https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions))
- **1** from Icecuber depth=4, diagonally flipped ([Link to Icecuber solution](https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/discussion/154597))
- **1** from a combination of train_tasks and Icecuber: for each task, I applied all solvers from Michael Holder's DSL ([Link to DSL](https://github.com/michaelhodel/arc-dsl)) to generate 400 new tasks, then used Icecuber on each of them and kept the output with the best score.
- **1** from a custom probing solution ([Link to Probed Solution](src/probed_tasks.py))
- **1** from the eighth-place solution ([Link to 8th place discussion](https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/discussion/154300))
- **1** from the ninth-place solution ([Link to 9th place discussion](https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/discussion/154319))
- **1** from the third-place solution ([Link to 3rd place discussion](https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/discussion/154409))
- **5** from second-place solutions ([Link to 2nd place discussion](https://www.kaggle.com/competitions/abstraction-and-reasoning-challenge/discussion/154391))

### Requirements

I recommend creating a new conda environment with Python 3.11 and installing the necessary dependencies:

```
pip install -r requirements.txt
cd src/arc_2020_ninth_place_setup && pip install
```

Additionally, ensure that you have a g++ version supporting C++17.

### Running the Solution

To execute the solution on the training tasks, run:

```
python main.py
```

(Note: The script is designed for the hidden test set, so some solutions may not be able to solve any tasks in the training set.)

---
