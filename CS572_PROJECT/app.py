import streamlit as st
import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# ==============================================================================
# --- STREAMLIT UI & SIDEBAR SLIDERS ---
# ==============================================================================
st.set_page_config(page_title="Mess Hall Simulator", layout="wide")
st.title("🍔 Campus Mess Hall Simulation Visualizer")

st.sidebar.header("⚙️ Simulation Controls")

# 1. Facility Sliders
st.sidebar.subheader("1. Facility Constraints")
NUM_MC_RUNS = st.sidebar.slider("Monte Carlo Runs", min_value=1, max_value=100, value=25)
COUNTER_CAPACITY = st.sidebar.slider("Serving Counters", 1, 30, 12)
SEATING_CAPACITY = st.sidebar.slider("Seating Capacity", 50, 500, 330)
SERVE_TIME_MIN = st.sidebar.slider("Min Serve Time (mins)", 0.1, 1.0, 0.75)
SERVE_TIME_MAX = st.sidebar.slider("Max Serve Time (mins)", 1.0, 3.0, 1.5)

# 2. Behavioral Sliders
st.sidebar.subheader("2. Psychology & Behavior")
PROB_SOCIAL = st.sidebar.slider("Probability of Social Groups", 0.0, 1.0, 0.8)
FATIGUE = st.sidebar.slider("Campus Fatigue Factor", 0.0, 1.0, 0.0)
MENU_MULT = st.sidebar.slider("Menu Quality Multiplier", 0.5, 1.5, 1.0)
PROB_NOT_EAT_IF_LATE = st.sidebar.slider("Base Prob Skip if Late", 0.0, 1.0, 0.8)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Procrastination Curve (Beta)")
WAKE_BETA_ALPHA = st.sidebar.slider("Beta Dist Alpha (Wake)", 1.0, 10.0, 5.00)
WAKE_BETA_BETA = st.sidebar.slider("Beta Dist Beta (Wake)", 1.0, 10.0, 1.50)

# 3. Dynamic Timetable Toggles
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Academic Timetable Setup")
st.sidebar.caption("Assign 8 AM, 9 AM, Both, or No Classes to each branch.")

YEARS = ["1st_Year", "2nd_Year", "3rd_Year", "4th_Year"]
BRANCHES = ["MnC", "CS", "ME", "EE"]
TIMETABLE = {}

for year in YEARS:
    with st.sidebar.expander(f"🎓 {year.replace('_', ' ')}"):
        for branch in BRANCHES:
            default_classes = []
            if year == "1st_Year" and branch in ["CS", "EE"]: default_classes = ["8 AM"]
            elif year == "2nd_Year" and branch == "ME": default_classes = ["8 AM"]
            elif year == "3rd_Year" and branch == "ME": default_classes = ["8 AM"]
            
            selected = st.multiselect(f"{branch} Classes", ["8 AM", "9 AM"], default=default_classes, key=f"{year}_{branch}")
            
            times = []
            if "8 AM" in selected: times.append(30)
            if "9 AM" in selected: times.append(90)
            TIMETABLE[f"{year}_{branch}"] = times

# Constants
MESS_CLOSING_TIME = 120      
SIM_END_TIME = 150           
STUDENTS_PER_BRANCH = 40
TOTAL_STUDENTS = len(YEARS) * len(BRANCHES) * STUDENTS_PER_BRANCH

# ==============================================================================
# --- LIVE BETA DISTRIBUTION VISUALIZER ---
# ==============================================================================
st.markdown("### 📊 Sleep Procrastination Curve (Beta Distribution)")
st.caption("This curve represents how students distribute their wake-up times relative to their SPECIFIC first class time. "
           "A steep curve on the right means they wake up at the last possible minute before their deadline.")

x = np.linspace(0, 1, 100)
x_safe = np.clip(x, 1e-5, 1 - 1e-5)
y = (x_safe**(WAKE_BETA_ALPHA-1)) * ((1-x_safe)**(WAKE_BETA_BETA-1))
y = y / np.max(y) 

fig_beta, ax_beta = plt.subplots(figsize=(12, 2.5))
ax_beta.plot(x, y, color='#9b59b6', linewidth=3)
ax_beta.fill_between(x, 0, y, color='#9b59b6', alpha=0.3)
ax_beta.set_yticks([])
ax_beta.set_xticks([0, 0.5, 1.0])
ax_beta.set_xticklabels(['Wakes Up Early', 'Wakes Up Midway', 'Wakes Up at Deadline'], fontweight='bold')
ax_beta.spines['top'].set_visible(False)
ax_beta.spines['right'].set_visible(False)
ax_beta.spines['left'].set_visible(False)
st.pyplot(fig_beta)

st.markdown("---")

# ==============================================================================
# --- SIMULATION ENGINE ---
# ==============================================================================

class MessHall:
    def __init__(self, env):
        self.env = env
        self.counters = simpy.Resource(env, capacity=COUNTER_CAPACITY)
        self.seats = simpy.Resource(env, capacity=SEATING_CAPACITY)
        self.arrival_data = []
        self.wait_data = []

    def serve_food(self):
        yield self.env.timeout(random.uniform(SERVE_TIME_MIN, SERVE_TIME_MAX))

    def provide_seat(self, eat_time):
        yield self.env.timeout(eat_time)

class FriendGroup:
    def __init__(self, env, group_id):
        self.env = env
        self.group_id = group_id
        self.members = []
        self.wake_event = env.event()   
        self.ready_event = env.event()  
        self.ready_count = 0
        
    def student_is_ready(self):
        self.ready_count += 1
        if self.ready_count == len(self.members):
            self.ready_event.succeed() 

class Student:
    def __init__(self, env, roll_no, year, branch, mess):
        self.env = env
        self.mess = mess
        self.batch = f"{year}_{branch}"
        self.friend_group = None 
        self.rushed_eating = False 
        self.has_eaten = False # <--- Tracks if they missed breakfast
        
        # Determine schedule
        self.class_schedule = TIMETABLE[self.batch]
        self.first_class = self.class_schedule[0] if len(self.class_schedule) > 0 else None
        self.last_class = self.class_schedule[-1] if len(self.class_schedule) > 0 else None
            
        fatigue_delay = FATIGUE * 60 
        menu_delay = (1.0 - MENU_MULT) * 40 
        self.prob_skip = max(0.0, min(1.0, PROB_NOT_EAT_IF_LATE + (1.0 - MENU_MULT) * 0.8))
            
        # Sleep Procrastination is multiplied by class time
        if self.first_class:
            raw_wake = random.betavariate(WAKE_BETA_ALPHA, WAKE_BETA_BETA) * self.first_class
        else:
            raw_wake = random.betavariate(WAKE_BETA_ALPHA, WAKE_BETA_BETA) * 150
            
        self.wake_up_time = max(0, raw_wake + fatigue_delay + menu_delay)

    def live_morning(self):
        if self.friend_group:
            natural_wake = self.env.timeout(self.wake_up_time)
            result = yield natural_wake | self.friend_group.wake_event
            if natural_wake in result and not self.friend_group.wake_event.triggered:
                self.friend_group.wake_event.succeed()
            yield self.env.timeout(random.uniform(10, 20))
            self.friend_group.student_is_ready()
            yield self.friend_group.ready_event 
        else:
            yield self.env.timeout(self.wake_up_time)
            yield self.env.timeout(random.uniform(10, 20))
            
        if self.first_class and (self.env.now > self.first_class):
            if random.random() <= self.prob_skip:
                class_end_time = self.last_class + random.uniform(50, 60)
                actual_return_time = class_end_time + max(6, random.gauss(10, 2.5))
                
                if actual_return_time <= MESS_CLOSING_TIME:
                    yield self.env.timeout(max(0, actual_return_time - self.env.now))
                    if self.env.now <= MESS_CLOSING_TIME:
                        yield self.env.process(self.visit_mess())
            else:
                self.rushed_eating = True
                if self.env.now <= MESS_CLOSING_TIME:
                    yield self.env.process(self.visit_mess())
        else:
            if self.env.now <= MESS_CLOSING_TIME:
                yield self.env.process(self.visit_mess())

    def visit_mess(self):
        arrive_time = self.env.now
        with self.mess.counters.request() as counter_req:
            yield counter_req  
            self.mess.wait_data.append(self.env.now - arrive_time)
            self.mess.arrival_data.append(arrive_time)
            yield self.env.process(self.mess.serve_food())
            
            # Successfully got food!
            self.has_eaten = True
            
        with self.mess.seats.request() as seat_req:
            yield seat_req  
            yield self.env.process(self.mess.provide_seat(random.uniform(10, 15) if self.rushed_eating else random.uniform(15, 30)))

# ==============================================================================
# --- OBSERVER (FOR LIVE VISUALIZATION) ---
# ==============================================================================
def mess_monitor(env, mess, history):
    while True:
        history.append({
            'minute': int(env.now),
            'queue_len': len(mess.counters.queue),
            'seats_used': mess.seats.count,
            'counters_used': mess.counters.count
        })
        yield env.timeout(1)

def setup_simulation(env, track_history=False):
    mess = MessHall(env)
    history = []
    if track_history:
        env.process(mess_monitor(env, mess, history))

    all_students = [] 
    roll_counter = 1
    for year in YEARS:
        for branch in BRANCHES:
            for _ in range(STUDENTS_PER_BRANCH):
                all_students.append(Student(env, f"R_{roll_counter}", year, branch, mess))
                roll_counter += 1
                
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
            
    for s in all_students: env.process(s.live_morning())
    return mess, history, all_students


# ==============================================================================
# --- DASHBOARD APP TABS ---
# ==============================================================================

tab1, tab2 = st.tabs(["🔴 Live Animation Visualizer", "📊 Monte Carlo Analyzer"])

# --- TAB 1: LIVE VISUALIZER ---
with tab1:
    st.markdown("### Watch the Queue Dynamics Unfold in Real-Time")
    if st.button("▶️ Start Live 1-Day Simulation", type="primary"):
        col1, col2, col3 = st.columns(3)
        time_metric = col1.metric("Current Time", "7:30 AM")
        queue_metric = col2.metric("Students Waiting in Line", "0")
        seat_metric = col3.metric("Seats Occupied", "0 / " + str(SEATING_CAPACITY))
        
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        
        env = simpy.Environment()
        mess, history, all_students = setup_simulation(env, track_history=True)
        env.run(until=SIM_END_TIME)
        
        df_hist = pd.DataFrame(history)
        
        for i in range(1, len(df_hist)):
            current_data = df_hist.iloc[:i]
            current_row = current_data.iloc[-1]
            
            clock_time = current_row['minute']
            hours = 7 + (clock_time + 30) // 60
            mins = (clock_time + 30) % 60
            formatted_time = f"{int(hours):02d}:{int(mins):02d} {'AM' if hours < 12 else 'PM'}"
            if clock_time >= 120: formatted_time = "🔒 DOORS LOCKED"
            
            time_metric.metric("Current Time", formatted_time)
            queue_metric.metric("Students Waiting in Line", f"{int(current_row['queue_len'])}")
            seat_metric.metric("Seats Occupied", f"{int(current_row['seats_used'])} / {SEATING_CAPACITY}")
            progress_bar.progress(min(1.0, current_row['minute'] / SIM_END_TIME))
            
            chart_placeholder.line_chart(
                data=current_data.set_index('minute')[['queue_len', 'seats_used']], 
                use_container_width=True,
                color=["#e74c3c", "#3498db"]
            )
            time.sleep(0.05) 
            
        # Display Final Total
        st.success(f"🏁 **Simulation Complete!** Total students who successfully ate breakfast: **{len(mess.arrival_data)} / {TOTAL_STUDENTS}**")
        
        # Display Missed Breakdown Matrix
        st.markdown("#### ❌ Breakfast Misses by Batch")
        st.caption("Number of students per batch who overslept or missed the mess closing time (Total 40 students per batch).")
        
        missed_matrix = np.zeros((4, 4), dtype=int)
        for i, year in enumerate(YEARS):
            for j, branch in enumerate(BRANCHES):
                batch_name = f"{year}_{branch}"
                misses = sum(1 for s in all_students if s.batch == batch_name and not s.has_eaten)
                missed_matrix[i, j] = misses
                
        df_missed = pd.DataFrame(missed_matrix, index=[y.replace("_", " ") for y in YEARS], columns=BRANCHES)
        
        # Apply a red heat-map style strictly to the missed dataframe
        st.dataframe(df_missed.style.background_gradient(cmap='Reds', axis=None), use_container_width=True)

# --- TAB 2: MONTE CARLO ---
with tab2:
    st.markdown(f"### Run {NUM_MC_RUNS} Simulations to Find the Expected Average")
    if st.button("📊 Run Monte Carlo Analysis"):
        with st.spinner(f"Running {NUM_MC_RUNS} simulated mornings..."):
            all_runs = []
            total_arrivals_list = []
            mc_missed_matrix = np.zeros((4, 4), dtype=float)
            
            for run in range(NUM_MC_RUNS):
                env = simpy.Environment()
                mess, _, all_students = setup_simulation(env)
                env.run(until=SIM_END_TIME)
                
                total_arrivals_list.append(len(mess.arrival_data))
                
                # Tally up misses for this run
                for i, year in enumerate(YEARS):
                    for j, branch in enumerate(BRANCHES):
                        batch_name = f"{year}_{branch}"
                        misses = sum(1 for s in all_students if s.batch == batch_name and not s.has_eaten)
                        mc_missed_matrix[i, j] += misses
                
                df_run = pd.DataFrame({'Arrival_Time': mess.arrival_data})
                if len(df_run) > 0:
                    df_run['Time_Window'] = pd.cut(df_run['Arrival_Time'], bins=range(0, 122, 1), labels=[str(i) for i in range(0, 121)], right=False)
                    run_summary = df_run.groupby('Time_Window', observed=False).size().reset_index(name='Arrival_Count')
                    all_runs.append(run_summary)
            
            # Average out the missed matrix across all runs
            mc_missed_matrix = mc_missed_matrix / NUM_MC_RUNS
            avg_total_turnout = np.mean(total_arrivals_list)
            
            st.success(f"📊 **Average Total Turnout:** {avg_total_turnout:.1f} / {TOTAL_STUDENTS} students")
            
            # Draw Expected Arrivals Chart
            master_df = pd.concat(all_runs, ignore_index=True)
            master_df['Time_Window'] = master_df['Time_Window'].astype(int)
            final_summary = master_df.groupby('Time_Window').agg(Arrival_Mean=('Arrival_Count', 'mean')).reset_index()
            final_summary['Arrival_Mean'] = final_summary['Arrival_Mean'].rolling(window=3, min_periods=1).mean()
            
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(final_summary['Time_Window'], final_summary['Arrival_Mean'], color='seagreen', linewidth=3)
            ax2.fill_between(final_summary['Time_Window'], 0, final_summary['Arrival_Mean'], color='seagreen', alpha=0.3)
            ax2.set_title(f"Expected Arrivals per Minute (Average of {NUM_MC_RUNS} runs)", fontweight="bold")
            ax2.set_xlabel("Minute of the Morning (0 = 7:30 AM)")
            ax2.set_ylabel("Average Students Arriving")
            
            ax2.axvline(x=MESS_CLOSING_TIME, color='black', linestyle='--', linewidth=2)
            ax2.text(MESS_CLOSING_TIME - 2, max(final_summary['Arrival_Mean']) * 0.8, 'Doors Lock', color='black', rotation=90, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
            # Display Average Missed Matrix
            st.markdown("#### ❌ Average Breakfast Misses by Batch")
            st.caption(f"Average number of students per batch who missed breakfast across {NUM_MC_RUNS} runs.")
            df_avg_missed = pd.DataFrame(np.round(mc_missed_matrix, 1), index=[y.replace("_", " ") for y in YEARS], columns=BRANCHES)
            st.dataframe(df_avg_missed.style.background_gradient(cmap='Reds', axis=None), use_container_width=True)