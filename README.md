# playground
Control Console that integrate Jovian (Virtual Reality Platform) and Spiketag (Real-time Ephys) 

Mouse and Rats or even human play in playground 
- It bidirectionally communicate with `Jovian` (through socket)
- It bidirectionally communicate `Spiketag` (through socket) 
- It bidirectionally communicate `user` (through low-latency rendering and interaction)  


### Functionality and Time budget
The `time budget` comes from the fact that the `playground` continuously receive VR stream from Jovian and feedback decision to Jovian every `frame`. If the frame rate is 60 then the time budget is 1/60=16.6ms. 
That means the following functionality needs to be done within this time budget
1. Real-time GPU rendering for trajectory/events visualization  (Navigation View)
2. User interaction (could be a script)   (A QT5 Console) 
3. Real-time Ephys signal visualization/interaction  (Extension of spiketag) 
4. Behaviour protocol FSM that takes interaction/Ephys as input  (Rule) 


To see this in a finer level we decompose it into 3 stages:

### Pre-stage processing
1. Parse the `Jovian` Maze created in `Blender` into `Mesh` object that can be rendered in playground
2. Parse the `Maze objects coordination` which will be used by `behaviour protocol FSM` 



### Main Loop processing (constrained by time budget)
1. Define `behave variable` (`Jovian stream` contains position, head direction, speed) 
2. Define `ephys variable` (`spiketag stream` contains ripple, spikes identity, theta etc. )
3. Define `object variable` which is by default `Maze objects coordination` (reword location, cue location, wall location) 
4. Define `rule` (A FSM receive inputs from `behave variable`, `ephys variable`, and `object variable` and issue `events`) 
5. Define `events` (issued by `rule` based on the interaction between `behave/ephys/object variables`) 
6. Define `score` (convert `events` into measurement of animal's performance) 

**Note:**
- `behave variable` and `object variable` interact to generate `events`
- `events` and `rule` interact to generate `measurement`
- `measurement` can influence `rule` to make it dynamic 
- `user` can top-down control `behave variable`, `object variable`, `rule` but not `events` and `score`



### Post-stage processing
1. Save the `trajectory file`
2. Save the `event log` 
3. Save the `measurement log` 
