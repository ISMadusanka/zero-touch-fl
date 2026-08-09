import re
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def parse_log(log_path):
    attack_success_dict = {}
    learn_dict = {}
    
    # Regexes to capture relevant fields
    # Example: [metrics.tracker] INFO: Metrics [round=22] ... goal_met=True evaded=True
    # `goal_met` is the attack GOAL (damage + collateral + evasion); the older
    # `attack_success=` spelling meant evasion alone and is still accepted so
    # logs from before the rename keep parsing.
    metrics_re = re.compile(
        r'Metrics \[round=(\d+)\].*?(?:goal_met|attack_success)=(True|False)')
    # Example: [rl.schedule] INFO: Round 22 [learn=attacker
    learn_re = re.compile(r'Round (\d+) \[learn=(attacker|defender)')
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = metrics_re.search(line)
            if m:
                rnd = int(m.group(1))
                attack_success_dict[rnd] = 1 if m.group(2) == 'True' else 0
                
            m_learn = learn_re.search(line)
            if m_learn:
                rnd = int(m_learn.group(1))
                learn_dict[rnd] = 1 if m_learn.group(2) == 'attacker' else 0

    all_rounds = sorted(list(set(list(attack_success_dict.keys()) + list(learn_dict.keys()))))
    
    x = []
    y_attack = []
    y_learn = []
    
    for r in all_rounds:
        x.append(r)
        y_attack.append(attack_success_dict.get(r, None))
        y_learn.append(learn_dict.get(r, None))
        
    return x, y_attack, y_learn

def main():
    log_file = r'c:\fl\server\github\zero-touch-fl\logs\system.log'
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return

    print("Parsing log file...")
    x, y_attack, y_learn = parse_log(log_file)
    
    if not x:
        print("No valid data found in the log.")
        return
        
    print(f"Extracted data for {len(x)} rounds.")
    print("Generating plot...")

    # Create subplot with 2 rows that share the x-axis for zooming
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Attack Success (True/False)', 'Learning Phase (Attacker/Defender)'),
                        vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=x, y=y_attack, mode='lines+markers', name='Attack Success', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y_learn, mode='lines+markers', name='Learner', line=dict(color='blue', shape='hv')), row=2, col=1)

    fig.update_layout(
        title_text="Zero-Touch FL System Log Visualization", 
        height=800, 
        hovermode="x unified",
        template="plotly_dark"
    )
    
    fig.update_xaxes(title_text="Round Number", row=2, col=1)
    
    # Update y-axes ticks for clarity
    fig.update_yaxes(tickvals=[0, 1], ticktext=['False', 'True'], row=1, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=['Defender', 'Attacker'], row=2, col=1)

    # Save to HTML
    output_html = 'visualization.html'
    fig.write_html(output_html)
    print(f"Success! Interactive plot generated and saved to {output_html}.")
    print(f"You can open {os.path.abspath(output_html)} in any web browser to view, zoom and pan the graphs.")
    
    # Attempt to open automatically if running locally
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(output_html))
    except Exception as e:
        pass

if __name__ == '__main__':
    main()
