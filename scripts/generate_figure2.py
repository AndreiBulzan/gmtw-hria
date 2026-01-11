#!/usr/bin/env python3
"""
Generate Figure 2: Task World Examples (2x2 grid)
Publication-quality figure for GMTW-Ro paper

Design: Minimal, academic, professional
Palette: Grayscale with subtle blue accent
Data: Accurate to the actual codebase
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.font_manager as fm

# Professional academic font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5

# Minimal color palette - grayscale with one accent
COLORS = {
    # Primary
    'primary': '#1a1a2e',       # Near-black for headers
    'accent': '#2563eb',        # Subtle blue accent

    # Grays
    'text': '#1f2937',          # Dark gray text
    'text_secondary': '#6b7280', # Medium gray
    'text_muted': '#9ca3af',    # Light gray

    # Backgrounds
    'bg': '#ffffff',            # White
    'bg_subtle': '#f9fafb',     # Very light gray
    'bg_card': '#f3f4f6',       # Light gray cards

    # Borders
    'border': '#e5e7eb',        # Light border
    'border_dark': '#d1d5db',   # Darker border

    # Semantic (muted)
    'success': '#059669',       # Muted green
    'error': '#dc2626',         # Muted red
    'priority_high': '#b45309', # Amber/brown for high priority
}


def add_panel_header(ax, title, subtitle=None):
    """Add a minimal header bar"""
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.89), 0.96, 0.09,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        facecolor=COLORS['primary'], edgecolor='none',
        transform=ax.transAxes
    ))

    if subtitle:
        ax.text(0.5, 0.935, title, transform=ax.transAxes,
                fontsize=9, fontweight='bold', color='white',
                ha='center', va='center')
        ax.text(0.5, 0.905, subtitle, transform=ax.transAxes,
                fontsize=7, color='#a0a0a0',
                ha='center', va='center')
    else:
        ax.text(0.5, 0.935, title, transform=ax.transAxes,
                fontsize=9, fontweight='bold', color='white',
                ha='center', va='center')


def add_constraint_pill(ax, text, x, y, width=None):
    """Add a constraint pill with auto-sizing"""
    if width is None:
        width = max(0.14, len(text) * 0.0115 + 0.04)

    half_w = width / 2

    ax.add_patch(FancyBboxPatch(
        (x - half_w, y - 0.022), width, 0.044,
        boxstyle="round,pad=0.003,rounding_size=0.015",
        facecolor=COLORS['bg_card'], edgecolor=COLORS['border'],
        transform=ax.transAxes, linewidth=0.5
    ))
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=5.5, color=COLORS['text'], ha='center', va='center')


def add_section_label(ax, text, x, y):
    """Add a small section label"""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=7, fontweight='medium', color=COLORS['text'])


def draw_travel_world(ax):
    """Panel A: Travel World - Based on actual codebase data"""
    add_panel_header(ax, "Travel World", "Brașov")

    add_section_label(ax, "Available attractions:", 0.04, 0.82)

    # From travel.py - actual attraction data
    attractions = [
        ("Biserica Neagră", "monument, interior", "25 lei"),
        ("Parcul Central", "parc, exterior", "0 lei"),
        ("Muzeul de Istorie", "muzeu, interior", "20 lei"),
        ("Telecabina Tâmpa", "transport, exterior", "22 lei"),
    ]

    # Table header line
    ax.plot([0.04, 0.96], [0.78, 0.78], color=COLORS['border'],
            linewidth=0.5, transform=ax.transAxes)

    for i, (name, attrs, cost) in enumerate(attractions):
        y = 0.73 - i * 0.072

        if i % 2 == 0:
            ax.add_patch(Rectangle(
                (0.04, y - 0.028), 0.92, 0.062,
                facecolor=COLORS['bg_subtle'], edgecolor='none',
                transform=ax.transAxes
            ))

        ax.text(0.06, y, name, transform=ax.transAxes,
                fontsize=6.5, color=COLORS['text'])
        ax.text(0.52, y, attrs, transform=ax.transAxes,
                fontsize=5.5, color=COLORS['text_secondary'])
        ax.text(0.94, y, cost, transform=ax.transAxes,
                fontsize=5.5, color=COLORS['accent'], ha='right',
                fontweight='medium')

    # Constraints section - from actual constraint types in travel.py
    add_section_label(ax, "Constraints:", 0.04, 0.40)

    add_constraint_pill(ax, "Budget ≤ 60 lei", 0.18, 0.32, 0.22)
    add_constraint_pill(ax, "Include monument", 0.46, 0.32, 0.22)
    add_constraint_pill(ax, "Max 1 outdoor/day", 0.76, 0.32, 0.24)

    # Output preview
    ax.add_patch(FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.20,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        facecolor=COLORS['bg_subtle'], edgecolor=COLORS['border'],
        transform=ax.transAxes, linewidth=0.5
    ))
    ax.text(0.07, 0.20, "Output:", transform=ax.transAxes,
            fontsize=5.5, color=COLORS['text_muted'])
    ax.text(0.07, 0.14, '{"day1": ["Biserica Neagră"],',
            transform=ax.transAxes,
            fontsize=5.5, family='monospace', color=COLORS['text'])
    ax.text(0.07, 0.085, ' "day2": ["Muzeul de Istorie"]}',
            transform=ax.transAxes,
            fontsize=5.5, family='monospace', color=COLORS['text'])


def draw_schedule_world(ax):
    """Panel B: Schedule World - Based on actual codebase data"""
    add_panel_header(ax, "Schedule World")

    # From schedule.py - actual Romanian day/slot names
    days = ["Luni", "Marți", "Miercuri"]
    slots = ["dim.", "d-a."]  # dimineață, după-amiază abbreviated

    cal_left, cal_top = 0.14, 0.82
    cal_width, cal_height = 0.80, 0.34
    cell_w = cal_width / 3
    cell_h = cal_height / 2

    # Day headers
    for i, day in enumerate(days):
        ax.text(cal_left + cell_w * (i + 0.5), cal_top + 0.025,
                day, transform=ax.transAxes, fontsize=6.5, ha='center',
                fontweight='medium', color=COLORS['text'])

    # Slot labels
    for j, slot in enumerate(slots):
        y = cal_top - cell_h * (j + 0.5)
        ax.text(cal_left - 0.025, y,
                slot, transform=ax.transAxes, fontsize=5.5, ha='right',
                va='center', color=COLORS['text_secondary'])

    # Grid cells
    for col in range(3):
        for row in range(2):
            x = cal_left + cell_w * col
            y = cal_top - cell_h * (row + 1)
            ax.add_patch(Rectangle(
                (x, y), cell_w, cell_h,
                fill=False, edgecolor=COLORS['border'], linewidth=0.5,
                transform=ax.transAxes
            ))

    # Appointments with priority - from actual MEETING_TYPES in schedule.py
    appointments = [
        (0, 0, "Ședință proiect", True),    # High priority
        (1, 0, "Prezentare", True),
        (0, 1, "Antrenament", False),
        (2, 1, "Review", False),
    ]

    for col, row, name, is_high in appointments:
        x = cal_left + cell_w * col
        y = cal_top - cell_h * (row + 1)

        bg_color = '#fef3c7' if is_high else COLORS['bg_subtle']
        ax.add_patch(Rectangle(
            (x + 0.004, y + 0.004), cell_w - 0.008, cell_h - 0.008,
            facecolor=bg_color, edgecolor='none',
            transform=ax.transAxes
        ))

        bar_color = COLORS['priority_high'] if is_high else COLORS['border_dark']
        ax.add_patch(Rectangle(
            (x + 0.004, y + 0.004), 0.01, cell_h - 0.008,
            facecolor=bar_color, edgecolor='none',
            transform=ax.transAxes
        ))

        ax.text(x + cell_w/2 + 0.008, y + cell_h/2, name,
                transform=ax.transAxes, fontsize=5.5, ha='center', va='center',
                color=COLORS['text'])

    # Constraints - from actual constraint types
    add_section_label(ax, "Constraints:", 0.04, 0.37)
    add_constraint_pill(ax, "Keep high priority", 0.22, 0.29, 0.28)
    add_constraint_pill(ax, "Max 2/day", 0.52, 0.29, 0.18)

    # Legend
    ax.add_patch(Rectangle((0.14, 0.08), 0.022, 0.022,
                            facecolor=COLORS['priority_high'], edgecolor='none',
                            transform=ax.transAxes))
    ax.text(0.175, 0.09, "High", transform=ax.transAxes,
            fontsize=5.5, va='center', color=COLORS['text_secondary'])

    ax.add_patch(Rectangle((0.30, 0.08), 0.022, 0.022,
                            facecolor=COLORS['border_dark'], edgecolor='none',
                            transform=ax.transAxes))
    ax.text(0.335, 0.09, "Normal", transform=ax.transAxes,
            fontsize=5.5, va='center', color=COLORS['text_secondary'])


def draw_fact_world(ax):
    """Panel C: Fact World - Misbelief trap from actual codebase"""
    add_panel_header(ax, "Fact World", "Misbelief Trap")

    # Context box - based on actual fact.py misbelief examples
    ax.add_patch(FancyBboxPatch(
        (0.04, 0.56), 0.92, 0.24,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=COLORS['bg_subtle'], edgecolor=COLORS['border'],
        transform=ax.transAxes, linewidth=0.5
    ))

    ax.text(0.08, 0.75, "Context provided:", transform=ax.transAxes,
            fontsize=6, color=COLORS['text_muted'])
    # Actual misbelief from fact.py: capital_romania has misbelief_answer="Sibiu"
    ax.text(0.50, 0.65, "Capitala României: Sibiu", transform=ax.transAxes,
            fontsize=7.5, ha='center', color=COLORS['text'], fontweight='medium',
            style='italic')  # Italic to show it's Romanian content
    ax.text(0.50, 0.595, "(deliberately incorrect)", transform=ax.transAxes,
            fontsize=5.5, ha='center', color=COLORS['text_muted'], style='italic')

    # Question - actual question_ro from fact.py
    ax.text(0.50, 0.49, "Q: Care este capitala României?",
            transform=ax.transAxes, fontsize=7, ha='center',
            color=COLORS['text'], fontweight='medium', style='italic')

    # Two answer boxes
    box_width = 0.40
    box_height = 0.22

    # Correct answer (follows context)
    ax.add_patch(FancyBboxPatch(
        (0.06, 0.15), box_width, box_height,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor='#f0fdf4', edgecolor=COLORS['success'],
        transform=ax.transAxes, linewidth=1
    ))
    ax.text(0.26, 0.33, "CORRECT", transform=ax.transAxes,
            fontsize=6, ha='center', color=COLORS['success'],
            fontweight='bold')
    ax.text(0.26, 0.255, '"Sibiu"', transform=ax.transAxes,
            fontsize=7.5, ha='center', color=COLORS['text'], fontweight='medium')
    ax.text(0.26, 0.185, "(follows context)", transform=ax.transAxes,
            fontsize=5.5, ha='center', color=COLORS['text_muted'])

    # Wrong answer (uses parametric knowledge)
    ax.add_patch(FancyBboxPatch(
        (0.54, 0.15), box_width, box_height,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor='#fef2f2', edgecolor=COLORS['error'],
        transform=ax.transAxes, linewidth=1
    ))
    ax.text(0.74, 0.33, "WRONG", transform=ax.transAxes,
            fontsize=6, ha='center', color=COLORS['error'],
            fontweight='bold')
    ax.text(0.74, 0.255, '"București"', transform=ax.transAxes,
            fontsize=7.5, ha='center', color=COLORS['text'], fontweight='medium')
    ax.text(0.74, 0.185, "(world knowledge)", transform=ax.transAxes,
            fontsize=5.5, ha='center', color=COLORS['text_muted'])

    # Bottom note
    ax.text(0.50, 0.055, "Tests context adherence vs. parametric knowledge",
            transform=ax.transAxes, fontsize=5.5, ha='center',
            color=COLORS['text_muted'], style='italic')


def draw_recipe_world(ax):
    """Panel D: Recipe World - Based on actual dishes from recipe.py"""
    add_panel_header(ax, "Recipe World", "Menu Planning")

    add_section_label(ax, "Available dishes:", 0.04, 0.82)

    # From recipe.py - actual Romanian dish data
    dishes = [
        ("Ciorbă de legume", "vegan", "200 kcal", True),
        ("Sarmale în foi de viță", "carne", "450 kcal", False),
        ("Mămăligă cu brânză", "vegetarian", "400 kcal", True),
        ("Piept pui la grătar", "carne", "350 kcal", False),
    ]

    ax.plot([0.04, 0.96], [0.78, 0.78], color=COLORS['border'],
            linewidth=0.5, transform=ax.transAxes)

    for i, (name, diet, cal, is_veg) in enumerate(dishes):
        y = 0.72 - i * 0.095

        card_color = '#f0fdf4' if is_veg else COLORS['bg_subtle']
        ax.add_patch(FancyBboxPatch(
            (0.04, y - 0.032), 0.92, 0.075,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor=card_color, edgecolor=COLORS['border'],
            transform=ax.transAxes, linewidth=0.3
        ))

        ax.text(0.07, y, name, transform=ax.transAxes,
                fontsize=6.5, color=COLORS['text'])

        tag_color = COLORS['success'] if is_veg else COLORS['text_secondary']
        ax.text(0.58, y, diet, transform=ax.transAxes,
                fontsize=5.5, color=tag_color)

        ax.text(0.92, y, cal, transform=ax.transAxes,
                fontsize=5.5, color=COLORS['text_secondary'], ha='right')

    # Constraints - from actual constraint types in recipe.py
    add_section_label(ax, "Constraints:", 0.04, 0.28)
    add_constraint_pill(ax, "Vegetarian", 0.15, 0.20, 0.18)
    add_constraint_pill(ax, "Max 1500 kcal/day", 0.42, 0.20, 0.28)
    add_constraint_pill(ax, "No repeats", 0.70, 0.20, 0.20)

    # Bottom hint
    ax.text(0.50, 0.055, "Model selects dishes satisfying all constraints",
            transform=ax.transAxes, fontsize=5.5, ha='center',
            color=COLORS['text_muted'], style='italic')


def main():
    # Create figure - compact
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    fig.patch.set_facecolor('white')

    # Configure axes
    for ax in axes.flat:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor(COLORS['bg'])

    # Draw panels
    draw_travel_world(axes[0, 0])
    draw_schedule_world(axes[0, 1])
    draw_fact_world(axes[1, 0])
    draw_recipe_world(axes[1, 1])

    # Panel labels
    for ax, label in zip(axes.flat, ['(a)', '(b)', '(c)', '(d)']):
        ax.text(0.02, 0.97, label, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', color=COLORS['text'])

    plt.tight_layout(pad=0.3)

    # Save
    output_path = 'docs/figure2_worlds.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    # PDF for publication
    plt.savefig('docs/figure2_worlds.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: docs/figure2_worlds.pdf")

    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
