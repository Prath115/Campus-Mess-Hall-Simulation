import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# ==============================================================================
# --- CONSTANTS & DATA STRUCTURES ---
# ==============================================================================
MEALS = {
    'Breakfast': {'start': 0, 'end': 120},    # 7:30 AM - 9:30 AM
    'Lunch': {'start': 300, 'end': 405},      # 12:30 PM - 2:15 PM
    'Snacks': {'start': 570, 'end': 645},     # 5:00 PM - 6:15 PM
    'Dinner': {'start': 720, 'end': 840}      # 7:30 PM - 9:30 PM
}
SIM_END_TIME = 850
YEARS = ["1st_Year", "2nd_Year", "3rd_Year", "4th_Year"]
BRANCHES = ["CS", "ME", "EE", "MnC"]

# ==============================================================================
# --- DATA PARSING BRIDGE ---
# ==============================================================================
def parse_real_timetable(file, target_day):
    """Extracts Morning, Midday, and Afternoon courses for the selected day."""
    df = pd.read_csv(file, header=None)
    current_day = None
    courses = {'8am': set(), '9am': set(), '12pm': set(), '2pm': set(), 'late_afternoon': set()}
    
    for index, row in df.iterrows():
        cell_0 = str(row[0]).strip()
        if cell_0 in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            current_day = cell_0
            
        if current_day == target_day:
            cols = {'8am': 2, '9am': 3, '12pm': 6, '2pm': 8, 'late_afternoon': 10} 
            for key, col_idx in cols.items():
                if col_idx < len(row):
                    val = str(row[col_idx])
                    if val != 'nan' and val.strip():
                        for c in val.replace('"', '').split(','):
                            courses[key].add(c.strip())
                            
            for col_idx in [12, 14]: # 4 PM and 5 PM classes
                if col_idx < len(row):
                    val = str(row[col_idx])
                    if val != 'nan' and val.strip():
                        for c in val.replace('"', '').split(','):
                            courses['late_afternoon'].add(c.strip())
    return courses

# ==============================================================================
# --- STREAMLIT UI & SIDEBAR ---
# ==============================================================================
st.set_page_config(page_title="Predictive Mess Hall Simulator", layout="wide")
st.title("🍔 Full Day Mess Hall Digital Twin & Catering Predictor")

st.sidebar.header("⚙️ Simulation Controls")

# --- SECTION 1: Data Import ---
with st.sidebar.expander("📂 1. Campus Data & Schedule", expanded=True):
    timetable_file = st.file_uploader("Upload Classroom Allocation (CSV)", type=['csv'])
    student_file = st.file_uploader("Upload Students Registry (CSV)", type=['csv'])
    selected_day = st.selectbox("📅 Select Day to Simulate", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

# --- SECTION 2: Facility Constraints ---
with st.sidebar.expander("🏢 2. Facility Constraints", expanded=False):
    COUNTER_CAPACITY = st.slider("Serving Counters", 1, 30, 12)
    SEATING_CAPACITY = st.slider("Seating Capacity", 50, 500, 330)

# --- SECTION 3: Student Psychology ---
with st.sidebar.expander("🧠 3. Student Psychology", expanded=False):
    PROB_SOCIAL = st.slider("Probability of Social Groups", 0.0, 1.0, 0.8, help="How many students arrive with friends.")
    FATIGUE = st.slider("Campus Fatigue Factor", 0.0, 1.0, 0.0, help="Higher fatigue means students wake up later and skip snacks.")
    MENU_MULT = st.slider("Menu Quality Multiplier", 0.5, 1.5, 1.0, help="0.5 = Awful Food (High Skips), 1.5 = Great Food (Low Skips)")
    PROB_NOT_EAT_IF_LATE = st.slider("Base Prob Skip if Late", 0.0, 1.0, 0.8)

# --- SECTION 4: Math & Distributions ---
with st.sidebar.expander("📈 4. Math & Distributions", expanded=False):
    WAKE_BETA_ALPHA = st.slider("Beta Dist Alpha (Wake)", 1.0, 10.0, 5.00)
    WAKE_BETA_BETA = st.slider("Beta Dist Beta (Wake)", 1.0, 10.0, 1.50)

# --- SECTION 5: Monte Carlo Settings ---
with st.sidebar.expander("🎲 5. Monte Carlo Settings", expanded=True):
    NUM_MC_RUNS = st.slider("Monte Carlo Runs", min_value=10, max_value=1000, value=50, step=10)
    CONFIDENCE_LEVEL = st.slider("Confidence Interval Level (%)", min_value=50, max_value=99, value=95, step=1, help="Adjust the width of the confidence band around the expected average.")

# ==============================================================================
# --- SIMULATION ENGINE ---
# ==============================================================================

class MessHall:
    def __init__(self, env):
        self.env = env
        self.counters = simpy.Resource(env, capacity=COUNTER_CAPACITY)
        self.seats = simpy.Resource(env, capacity=SEATING_CAPACITY)
        self.arrivals = {'Breakfast': [], 'Lunch': [], 'Snacks': [], 'Dinner': []}

    def serve_food(self):
        yield self.env.timeout(random.uniform(0.75, 1.5))

    def provide_seat(self, eat_time):
        yield self.env.timeout(eat_time)

class FriendGroup:
    def __init__(self, env, group_id):
        self.env = env
        self.members = []
        self.dinner_sync_event = env.event()   
        self.ready_count = 0
        
    def dinner_is_ready(self):
        self.ready_count += 1
        if self.ready_count == len(self.members):
            self.dinner_sync_event.succeed() 

class Student:
    def __init__(self, env, roll_no, batch, mess, daily_schedule):
        self.env = env
        self.mess = mess
        self.batch = batch
        self.roll_no = roll_no
        self.friend_group = None 
        self.daily_schedule = daily_schedule
        self.eaten = {'Breakfast': False, 'Lunch': False, 'Snacks': False, 'Dinner': False}
        
        fatigue_delay = FATIGUE * 60 
        self.menu_delay = (1.0 - MENU_MULT) * 40 
        self.prob_skip = max(0.0, min(1.0, PROB_NOT_EAT_IF_LATE + (1.0 - MENU_MULT) * 0.8))

        first_class = 30 if daily_schedule['has_8am'] else (90 if daily_schedule['has_9am'] else 150)
        raw_wake = random.betavariate(WAKE_BETA_ALPHA, WAKE_BETA_BETA) * first_class
        self.wake_up_time = max(0, raw_wake + fatigue_delay + self.menu_delay)

        release_time = 320 if daily_schedule['has_12pm'] else 300 
        deadline_time = 380 if daily_schedule['has_2pm'] else 405 
        self.lunch_arrive_time = random.uniform(release_time, deadline_time)

        base_snack_time = 585 if daily_schedule['has_late'] else 570
        self.snacks_arrive_time = base_snack_time + random.gammavariate(2, 10)
        
        self.dinner_arrive_time = random.gauss(780, 25) 

    def live_day(self):
        # BREAKFAST
        yield self.env.timeout(self.wake_up_time)
        yield self.env.timeout(random.uniform(10, 20)) 
        if self.env.now <= MEALS['Breakfast']['end']:
            if self.daily_schedule['has_8am'] and self.env.now > 30:
                if random.random() > self.prob_skip:
                    yield self.env.process(self.visit_mess('Breakfast', rushed=True))
            else:
                yield self.env.process(self.visit_mess('Breakfast', rushed=False))

        # LUNCH
        time_to_lunch = max(0, self.lunch_arrive_time - self.env.now)
        yield self.env.timeout(time_to_lunch)
        skip_lunch_chance = 0.10 + ((1.0 - MENU_MULT) * 0.3)
        if random.random() > skip_lunch_chance and self.env.now <= MEALS['Lunch']['end']:
            yield self.env.process(self.visit_mess('Lunch', rushed=self.daily_schedule['has_2pm']))

        # SNACKS
        time_to_snacks = max(0, self.snacks_arrive_time - self.env.now)
        yield self.env.timeout(time_to_snacks)
        skip_snacks_chance = 0.40 + (FATIGUE * 0.3) + ((1.0 - MENU_MULT) * 0.3)
        if random.random() > skip_snacks_chance and self.env.now <= MEALS['Snacks']['end']:
            yield self.env.process(self.visit_mess('Snacks', rushed=False))

        # DINNER
        time_to_dinner = max(0, self.dinner_arrive_time - self.env.now)
        yield self.env.timeout(time_to_dinner)
        if self.friend_group:
            self.friend_group.dinner_is_ready()
            yield self.friend_group.dinner_sync_event 
        skip_dinner_chance = 0.05 + ((1.0 - MENU_MULT) * 0.4) 
        if random.random() > skip_dinner_chance and self.env.now <= MEALS['Dinner']['end']:
            yield self.env.process(self.visit_mess('Dinner', rushed=False))

    def visit_mess(self, meal_name, rushed):
        arrive_time = self.env.now
        with self.mess.counters.request() as counter_req:
            yield counter_req  
            self.mess.arrivals[meal_name].append(arrive_time)
            yield self.env.process(self.mess.serve_food())
            self.eaten[meal_name] = True
            
        with self.mess.seats.request() as seat_req:
            yield seat_req  
            eat_time = random.uniform(10, 15) if rushed else random.uniform(15, 30)
            yield self.env.process(self.mess.provide_seat(eat_time))

def mess_monitor(env, mess, history):
    while True:
        history.append({
            'minute': int(env.now),
            'queue_len': len(mess.counters.queue),
            'seats_used': mess.seats.count
        })
        yield env.timeout(1)

def setup_simulation(env, df_students, daily_courses, track_history=False):
    mess = MessHall(env)
    history = []
    if track_history:
        env.process(mess_monitor(env, mess, history))

    all_students = [] 
    for index, row in df_students.iterrows():
        roll_no = str(row['Roll_Number'])
        batch = str(row['Year_Branch'])
        student_courses = [c.strip() for c in str(row['Courses_Registered']).split(',')]
        
        daily_schedule = {
            'has_8am': any(c in daily_courses['8am'] for c in student_courses),
            'has_9am': any(c in daily_courses['9am'] for c in student_courses),
            'has_12pm': any(c in daily_courses['12pm'] for c in student_courses),
            'has_2pm': any(c in daily_courses['2pm'] for c in student_courses),
            'has_late': any(c in daily_courses['late_afternoon'] for c in student_courses)
        }
        all_students.append(Student(env, roll_no, batch, mess, daily_schedule))
                
    social_candidates = [s for s in all_students if random.random() <= PROB_SOCIAL]
    unassigned = list(social_candidates)
    group_id = 1
    
    while unassigned:
        group = FriendGroup(env, f"G_{group_id}")
        seed = random.choice(unassigned)
        group.members.append(seed)
        unassigned.remove(seed)
        
        target = random.choice([3, 4])
        while len(group.members) < target and unassigned:
            friend = random.choice([s for s in unassigned if s.batch == seed.batch]) if random.random() <= 0.85 and any(s.batch == seed.batch for s in unassigned) else random.choice(unassigned)
            group.members.append(friend)
            unassigned.remove(friend)
            
        for m in group.members: m.friend_group = group
        group_id += 1
            
    for s in all_students: env.process(s.live_day())
    return mess, history, all_students

def minute_to_time(minute):
    hours = 7 + (minute + 30) // 60
    mins = (minute + 30) % 60
    am_pm = 'AM' if hours < 12 else 'PM'
    hours = hours if hours <= 12 else hours - 12
    return f"{int(hours):02d}:{int(mins):02d} {am_pm}"

# ==============================================================================
# --- MAIN APPLICATION DASHBOARD ---
# ==============================================================================

if not timetable_file or not student_file:
    st.warning("⚠️ Please upload both 'Classroom Allocation' and 'Students Registry' CSVs in the sidebar to begin.")
    st.stop()

daily_courses = parse_real_timetable(timetable_file, selected_day)
df_students = pd.read_csv(student_file)
TOTAL_STUDENTS = len(df_students)

st.success(f"✅ Data Loaded for **{selected_day}**. Simulating 14 hours for {TOTAL_STUDENTS} students.")

tab1, tab2 = st.tabs(["🔴 Live Animation Visualizer (1 Day)", "📊 Full Monte Carlo Analyzer"])

# --- TAB 1: LIVE VISUALIZER ---
with tab1:
    st.markdown("### Watch the Queue Dynamics Unfold Across the Entire Day")
    if st.button("▶️ Start Live 14-Hour Simulation", type="primary"):
        col1, col2, col3 = st.columns(3)
        time_metric = col1.metric("Current Time", "7:30 AM")
        queue_metric = col2.metric("Students Waiting in Line", "0")
        seat_metric = col3.metric("Seats Occupied", "0 / " + str(SEATING_CAPACITY))
        
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        
        env = simpy.Environment()
        mess, history, all_students = setup_simulation(env, df_students, daily_courses, track_history=True)
        env.run(until=SIM_END_TIME)
        
        df_hist = pd.DataFrame(history)
        
        for i in range(1, len(df_hist), 3): 
            current_data = df_hist.iloc[:i]
            current_row = current_data.iloc[-1]
            
            time_metric.metric("Current Time", minute_to_time(current_row['minute']))
            queue_metric.metric("Students Waiting in Line", f"{int(current_row['queue_len'])}")
            seat_metric.metric("Seats Occupied", f"{int(current_row['seats_used'])} / {SEATING_CAPACITY}")
            progress_bar.progress(min(1.0, current_row['minute'] / SIM_END_TIME))
            
            chart_placeholder.line_chart(
                data=current_data.set_index('minute')[['queue_len', 'seats_used']], 
                use_container_width=True,
                color=["#e74c3c", "#3498db"]
            )
            time.sleep(0.01) 
            
        st.success("🏁 **14-Hour Simulation Complete!** Review the Daily Catering Report below.")
        
        # --- SINGLE DAY CATERING REPORT ---
        st.markdown("---")
        st.markdown("### 📋 End of Day Catering Report")
        
        attendance_data = {meal: np.zeros((4, 4), dtype=int) for meal in MEALS.keys()}
        missed_data = {meal: np.zeros((4, 4), dtype=int) for meal in MEALS.keys()}
        
        for i, year in enumerate(YEARS):
            for j, branch in enumerate(BRANCHES):
                batch_name = f"{year}_{branch}"
                for meal in MEALS.keys():
                    attended = sum(1 for s in all_students if s.batch == batch_name and s.eaten[meal])
                    missed = sum(1 for s in all_students if s.batch == batch_name and not s.eaten[meal])
                    attendance_data[meal][i, j] = attended
                    missed_data[meal][i, j] = missed

        st.markdown("#### 🟢 Students Attending (Prepare Food)")
        cols_att = st.columns(4)
        for idx, meal in enumerate(MEALS.keys()):
            with cols_att[idx]:
                st.markdown(f"**{meal}**")
                df_att = pd.DataFrame(attendance_data[meal], index=[y.replace("_", " ") for y in YEARS], columns=BRANCHES)
                st.dataframe(df_att.style.background_gradient(cmap='Greens', axis=None), use_container_width=True)

        st.markdown("#### 🔴 Students Missing (Saved Food / Waste Prevented)")
        cols_miss = st.columns(4)
        for idx, meal in enumerate(MEALS.keys()):
            with cols_miss[idx]:
                st.markdown(f"**{meal}**")
                df_miss = pd.DataFrame(missed_data[meal], index=[y.replace("_", " ") for y in YEARS], columns=BRANCHES)
                st.dataframe(df_miss.style.background_gradient(cmap='Reds', axis=None), use_container_width=True)


# --- TAB 2: MONTE CARLO ---
with tab2:
    st.markdown(f"### Expected 14-Hour Traffic Pattern (Average of {NUM_MC_RUNS} runs at {CONFIDENCE_LEVEL}% Confidence)")
    if st.button("📊 Run Monte Carlo Analysis"):
        with st.spinner(f"Running {NUM_MC_RUNS} simulated days... Please wait."):
            
            time_bins = np.arange(0, SIM_END_TIME + 5, 5) 
            mc_arrivals_matrix = np.zeros((NUM_MC_RUNS, len(time_bins)-1))
            
            mc_attendance = {meal: np.zeros((4, 4), dtype=float) for meal in MEALS.keys()}
            mc_missed = {meal: np.zeros((4, 4), dtype=float) for meal in MEALS.keys()}
            
            for run in range(NUM_MC_RUNS):
                env = simpy.Environment()
                mess, _, all_students = setup_simulation(env, df_students, daily_courses)
                env.run(until=SIM_END_TIME)
                
                run_arrivals = []
                for meal, arrivals in mess.arrivals.items():
                    run_arrivals.extend(arrivals)
                    
                    for i, year in enumerate(YEARS):
                        for j, branch in enumerate(BRANCHES):
                            batch_name = f"{year}_{branch}"
                            attended = sum(1 for s in all_students if s.batch == batch_name and s.eaten[meal])
                            missed = sum(1 for s in all_students if s.batch == batch_name and not s.eaten[meal])
                            
                            mc_attendance[meal][i, j] += attended
                            mc_missed[meal][i, j] += missed
                
                counts, _ = np.histogram(run_arrivals, bins=time_bins)
                mc_arrivals_matrix[run, :] = counts
            
            # --- DYNAMIC CONFIDENCE INTERVAL MATH ---
            # Calculate the upper and lower percentile bounds based on the slider
            lower_bound_percentile = (100 - CONFIDENCE_LEVEL) / 2.0
            upper_bound_percentile = 100 - lower_bound_percentile
            
            arrival_mean = np.mean(mc_arrivals_matrix, axis=0) / 5.0
            arrival_lower = np.percentile(mc_arrivals_matrix, lower_bound_percentile, axis=0) / 5.0 
            arrival_upper = np.percentile(mc_arrivals_matrix, upper_bound_percentile, axis=0) / 5.0 
            minutes = time_bins[:-1]
            
            # --- PLOT GRAPH ---
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            
            # Plot the dynamic confidence interval shaded region
            ax2.fill_between(minutes, arrival_lower, arrival_upper, color='#3498db', alpha=0.3, label=f'{CONFIDENCE_LEVEL}% Confidence Interval')
            ax2.plot(minutes, arrival_mean, color='#2c3e50', linewidth=2, label='Expected Average')
            
            colors = ['#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
            for idx, (meal, times) in enumerate(MEALS.items()):
                ax2.axvspan(times['start'], times['end'], color=colors[idx], alpha=0.15)
                ax2.text(times['start'] + 10, ax2.get_ylim()[1]*0.9, meal, fontweight='bold', color=colors[idx])
            
            ticks = [0, 120, 300, 405, 570, 645, 720, 840]
            ax2.set_xticks(ticks)
            ax2.set_xticklabels([minute_to_time(t) for t in ticks], rotation=45, fontweight='bold')
            ax2.set_title(f"14-Hour Expected Arrivals ({NUM_MC_RUNS} Runs | {CONFIDENCE_LEVEL}% Confidence)", fontweight="bold", fontsize=14)
            ax2.set_ylabel("Students Arriving per Minute")
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
            # --- STATISTICAL CATERING REPORT ---
            st.markdown("---")
            st.markdown("### 📋 Statistical Predictive Catering (Averages)")
            
            st.markdown("#### 🟢 Expected Average Attendance")
            cols_att = st.columns(4)
            for idx, (meal, matrix) in enumerate(mc_attendance.items()):
                with cols_att[idx]:
                    st.markdown(f"**{meal}**")
                    avg_matrix = matrix / NUM_MC_RUNS
                    df_avg = pd.DataFrame(np.round(avg_matrix, 1), index=[y.replace("_", " ") for y in YEARS], columns=BRANCHES)
                    st.dataframe(df_avg.style.background_gradient(cmap='Greens', axis=None), use_container_width=True)

            st.markdown("#### 🔴 Expected Average Misses")
            cols_miss = st.columns(4)
            for idx, (meal, matrix) in enumerate(mc_missed.items()):
                with cols_miss[idx]:
                    st.markdown(f"**{meal}**")
                    avg_matrix = matrix / NUM_MC_RUNS
                    df_avg = pd.DataFrame(np.round(avg_matrix, 1), index=[y.replace("_", " ") for y in YEARS], columns=BRANCHES)
                    st.dataframe(df_avg.style.background_gradient(cmap='Reds', axis=None), use_container_width=True)