<div align="center">

  <h3 align="center">Connect 4 agent using MiniMax</h3>

  <p align="center">
    [WS 22/23] Programming Project in Python
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
As a part of the Programming Project in Python course held in WS22/23 at TU-Berlin we developed an agent to play the perfect information game Connect 4. 

It uses bitboards, the minimax algorithm, alpha-beta pruning, a transposition table and iterative deepening.



<!-- GETTING STARTED -->
## Getting Started

Download the project and run the main.py file locally using python. You may choose different ways to use the agent. Either against the user or against itself. To select, specify either
```
human_vs_agent(generate_move_minimax)
```
to let the agent play against a user or 
```
human_vs_agent(generate_move_minimax, generate_move_minimax)
```
to let the agent play against itself in the main method of the main.py file.

You can change the time available to the agent in the minimax.py file by changing the SECONDS_TO_PLAY variable.
```
SECONDS_TO_PLAY: int = 5
```

### Prerequisites
python-version
  * This project uses python version 3.10.

Using pip you are able to install the required packages quickly.
  ```sh
  pip install -r requirements.txt
  ```


<!-- CONTACT -->
## Contact

Ole Lenz - ole.lenz@gmx.net and ole.lenz@campus.tu-berlin.de

Stefan Warmboldt - stefan.warmboldt@gmx.net and stefan.warmboldt@campus.tu-berlin.de

Project Link: [https://github.com/olelenz/connect_four](https://github.com/olelenz/connect_four)


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

The following project by Pascal Pons implements a Connect 4 solver which was in this project used to adapt the evaluation function. Check it out if you want to play against a perfect agent.

* [GitHub project](https://github.com/PascalPons/connect4)
* [Solver](https://connect4.gamesolver.org/en/)
* [Explanations](http://blog.gamesolver.org/)
