# state_space

State space systems in rust, with update method.

There is a struct to hold the properties, for example the system matrices. An update() method is included to step the system forward in time. It uses forward-euler version of continuous equations of motion. The u, x, and y vectors can be bounded.
