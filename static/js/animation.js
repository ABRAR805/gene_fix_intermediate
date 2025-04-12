function startSimulation() {
    fetch('/live_simulate', {
      method: 'POST'
    })
      .then(response => response.json())
      .then(data => {
        const steps = data.steps;
        const container = document.getElementById('simulation-steps');
        container.innerHTML = '';
  
        let i = 0;
        const interval = setInterval(() => {
          if (i < steps.length) {
            const step = document.createElement("p");
            step.textContent = steps[i];
            step.classList.add("fade-in");
            container.appendChild(step);
            i++;
          } else {
            clearInterval(interval);
          }
        }, 1200);
      });
  }
  