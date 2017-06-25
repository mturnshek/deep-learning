document.addEventListener("DOMContentLoaded", function(event) { 
	// create an environment object
	var env = {};
	env.getNumStates = function() { return 8; }
	env.getMaxNumActions = function() { return 2; }

	// create the DQN agent
	var spec = { alpha: 0.01 } // see full options on DQN page
	agent = new RL.DQNAgent(env, spec); 

	setInterval(function(){ // start the learning loop
	  s = [0, 0, 0, 0, 0, 0, 0, 0];
	  var action = agent.act(s); // s is an array of length 8
	  console.log(action);
	  //... execute action in environment and get the reward
	  let reward = 0.0;
	  if (action == 0) {
	  	reward += 1.0;
	  }
	  agent.learn(reward); // the agent improves its Q,policy,model, etc. reward is a float
	}, 0);
});