// === Calendar Section ===
document.addEventListener('DOMContentLoaded', function () {
  var calendarEl = document.getElementById('calendar');
  if (calendarEl) {
    var calendar = new FullCalendar.Calendar(calendarEl, {
      initialView: 'dayGridMonth',
      headerToolbar: {
        left: 'prev,next today',
        center: 'title',
        right: ''
      },
      height: 'auto'
    });
    calendar.render();
  }
});

// === GPS Calendar Converter Section ===
const gpsEpoch = new Date(Date.UTC(1980, 0, 6)); // GPS Epoch = Jan 6, 1980

function gpsToGregorian(week, day) {
  const totalDays = (week * 7) + day;
  const resultDate = new Date(gpsEpoch);
  resultDate.setUTCDate(resultDate.getUTCDate() + totalDays);
  return resultDate.toISOString().split('T')[0];
}

function gregorianToGps(dateStr) {
  const inputDate = new Date(dateStr);
  const diffTime = inputDate.getTime() - gpsEpoch.getTime();
  const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
  const gpsWeek = Math.floor(diffDays / 7);
  const gpsDay = inputDate.getUTCDay(); // Sunday = 0
  return { gpsWeek, gpsDay };
}

// === Section Navigation Toggle ===
document.addEventListener("DOMContentLoaded", () => {
  const sections = document.querySelectorAll(".content-section");
  const navLinks = document.querySelectorAll("nav a[href^='#']");

  function showSection(id) {
    sections.forEach(sec => {
      sec.style.display = (sec.id === id) ? "block" : "none";
    });
  }

  // Initially show only home
  showSection("home");

  // Add click events to nav links
  navLinks.forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetId = link.getAttribute("href").replace("#", "");
      showSection(targetId);
    });
  });
});


// Populate the form inputs dynamically based on selection
document.addEventListener('DOMContentLoaded', function () {
  const conversionTypeSelect = document.getElementById("conversionType");
  const inputFields = document.getElementById("inputFields");
  const form = document.getElementById("converterForm");
  const resultOutput = document.getElementById("resultOutput");

  function updateInputFields() {
    const type = conversionTypeSelect.value;
    inputFields.innerHTML = "";

    if (type === "toGregorian") {
      inputFields.innerHTML = `
        <label>GPS Week: <input type="number" id="gpsWeek" required></label>
        <label>Day of Week (0=Sunday to 6=Saturday): <input type="number" id="gpsDay" min="0" max="6" required></label>
      `;
    } else {
      inputFields.innerHTML = `
        <label>Date (YYYY-MM-DD): <input type="date" id="gregorianDate" required></label>
      `;
    }
  }

  if (conversionTypeSelect) {
    conversionTypeSelect.addEventListener("change", updateInputFields);
    updateInputFields(); // Initialize on page load
  }

  if (form) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      const type = conversionTypeSelect.value;
      let result = "";

      if (type === "toGregorian") {
        const week = parseInt(document.getElementById("gpsWeek").value);
        const day = parseInt(document.getElementById("gpsDay").value);
        if (isNaN(week) || isNaN(day) || day < 0 || day > 6) {
          result = "Invalid input. Day must be between 0 and 6.";
        } else {
          result = `Gregorian Date: ${gpsToGregorian(week, day)}`;
        }
      } else {
        const dateStr = document.getElementById("gregorianDate").value;
        if (!dateStr) {
          result = "Please enter a valid date.";
        } else {
          const { gpsWeek, gpsDay } = gregorianToGps(dateStr);
          result = `GPS Week: ${gpsWeek}, Day: ${gpsDay}`;
        }
      }

      resultOutput.textContent = result;
    });
  }
});
// === Contact Form Submission (AJAX with transparent success message) ===
document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('contactForm');
  const feedback = document.getElementById('feedback');

  if (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      const formData = new FormData(form);

      fetch('send-mail.php', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (response.ok) {
          feedback.style.display = 'block';
          feedback.textContent = 'Message sent successfully!';
          form.reset();
        } else {
          feedback.style.display = 'block';
          feedback.textContent = 'Failed to send. Try again.';
        }
      })
      .catch(error => {
        feedback.style.display = 'block';
        feedback.textContent = 'An error occurred.';
      });
    });
  }
});
