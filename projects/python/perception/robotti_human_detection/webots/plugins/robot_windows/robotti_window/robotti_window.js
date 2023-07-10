import RobotWindow from 'https://cyberbotics.com/wwi/R2023b/RobotWindow.js';
/* eslint no-unused-vars: ['error', { 'varsIgnorePattern': 'handleBodyLEDCheckBox|toggleStopCheckbox' }] */

// Send instruction to add a new obstacle.
window.addObstacle =  function(obj) {
  const class_name = document.getElementById('add-class').value;
  const def_name = document.getElementById('add-def-name').value;
  const x = document.getElementById('add-position-x').value;
  const y = document.getElementById('add-position-y').value;
  console.log('add obstacle ' + def_name + ' ' + class_name + ' ' + x.toString() + ' ' + y.toString());
  window.robotWindow.send(`add obstacle ${class_name} ${def_name}  ${x.toString()} ${y.toString()}`);
}

// Remove obstacle from the scene.
window.removeObstacle = function(obj) {
  const def_name = document.getElementById('remove-def-name').value;
  if (def_name)
    window.robotWindow.send('remove obstacle ' + def_name);
  else
    console.error('Invalid obstacle DEF name to be removed.');
}

window.takePicture = function(obj) {
  window.robotWindow.send('shot');
}

// A message coming from the robot has been received.
function receive(message, robot) {
  if (message.startsWith('display-image')) {
    const imageElement = document.getElementById('robot-camera')
    if (imageElement)
      imageElement.setAttribute('src', message.substring(message.indexOf(':') + 1));
  } else if (message.startsWith('log')) {
    const ul = document.getElementById('console');
    let li = document.createElement('li');
    li.appendChild(document.createTextNode(message.substring(message.indexOf(':') + 1)));
    ul.appendChild(li);
    var elements = ul.getElementsByTagName('li');
    if (elements.length > 10)
      ul.removeChild(elements[0]);
  }
}

// Initialize the RobotWindow class in order to communicate with the robot.
window.onload = function() {
  console.log('HTML page loaded.');
  window.robotWindow = new RobotWindow();
  window.robotWindow.setTitle('Robotti Window');
  window.robotWindow.receive = receive;
};
