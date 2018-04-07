# playground
Control Console that integrate Jovian(Virtual Reality Platform) and Spiketag(Real-time Spikesorting) 

Mouse and Rats or even human play in playground 
It bidirectionally communicate with both Jovian/Spiketag and the player (You) 


### Functionality and Time budget
The `time budget` comes from the fact that the playground continuously receive VR stream from Jovian and feedback decision to Jovian every `frame`. If the frame rate is 60 then the time budget is 1/60=16.6ms. 
That means the following functionality needs to be done within this time budget
1. Real-time GPU rendering for trajectory/events visualization  
2. User interaction (could be a script) 
3. Real-time Ephys signal visualization/interaction 
4. Behaviour protocol FSM that takes interaction/Ephys as input 


To see this in a finer level we decompose it into 3 stages:

### Pre-stage processing
1. Parse the Maze into `Mesh` object that can be rendered in playground
2. Parse the `Maze objects coordination` which will be used by `behaviour protocol FSM` 



### Main Loop processing
1. Define `behave variable` (position, head direction, speed) 
2. Define `object variable` which is by default `Maze objects coordination` (reword location, cue location, wall location) 
3. Define `events` (issued by `rule`, triggered by `animal`, `user`, and `ephys`) 
4. Define `rule` (Which is a FSM receive input from `animal, `user and `ephys` and issue `events`) 
5. Define `score` (convert `events` into measurement of animal's performance) 

- `behave variable` and `object variable` interact to generate `events`
- `events` and `rule` interact to generate `measurement`
- `measurement` can influence `rule` to make it dynamic 
- `user` can top-down control `behave variable`, `object variable`, `rule` but not `events` and `score`



### Post-stage processing
1. Save the trajectory file
2. Save the event log 
3. Save the measurement log 
