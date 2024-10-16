# qosf mentorship task

In this repo I attempt a solution for Task 2: Noise, Noise, and More Noise.

I recommend going through the files in the following order:
1. Part_1.ipynb
2. Decompositions.ipynb
3. Part_2.ipynb
4. Part_3.ipynb
5. Part_4.ipynb


I demonstrate the approach for solving each part in separate files: Part_1, Part_2, and Part_3. Afterward, I combine all these steps in Part_4 to complete the screening task.

In Part_1, I take a list of pennylane operations, insert Pauli errors based on input probabilities and return a list of operations with the errors which can then be converted to a circuit that can be simulated.

In Part_2, I decompose the operations in a circuit into those in the target basis: ```CX```, ```ID```, ```RZ```, ```SX```, ```X```

In Part_3, I implement the Draper's QFT based adder to add two positive integers.

Finally, in Part_4 I put all the components together by first transpiling the adder circuit into the target basis and introducing noise.