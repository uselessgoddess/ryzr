use std::time::Instant;

use bevy::{
    ecs::hierarchy::ChildSpawnerCommands,
    prelude::*,
    window::{PresentMode, WindowResolution},
};
use ryzr_gui::{Board, CellKind, CellPos, EditorRuntime};

const CELL_SIZE: f32 = 38.0;
const CELL_GAP: f32 = 4.0;
const LEFT_PANEL_WIDTH: f32 = 210.0;
const RIGHT_PANEL_WIDTH: f32 = 300.0;
const MAX_TICKS_PER_UPDATE: u64 = 256;
const VIPER_CLOCK_TICKS_PER_CPU_CYCLE: u64 = 7;

#[derive(Component)]
struct CellButton {
    pos: CellPos,
}

#[derive(Component)]
struct CellLabel {
    pos: CellPos,
}

#[derive(Component)]
struct PaletteButton(CellKind);

#[derive(Component)]
struct ControlButton(ControlKind);

#[derive(Component)]
struct ControlButtonLabel(ControlKind);

#[derive(Component)]
struct MetricText(MetricKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlKind {
    Step,
    RunPause,
    Slower,
    Faster,
    Reset,
    ToggleInput,
    Demo,
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricKind {
    Status,
    Circuit,
    Selected,
    Sources,
    Leds,
    Vcb,
    Error,
}

#[derive(Resource)]
struct UiState {
    selected_tool: CellKind,
    selected_cell: Option<CellPos>,
    running: bool,
    ticks_per_update: u64,
    ticks_per_second: f64,
    last_tick_count: u64,
    last_sample: Instant,
    last_error: Option<String>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            selected_tool: CellKind::Wire,
            selected_cell: None,
            running: std::env::var("RYZR_GUI_AUTORUN").is_ok(),
            ticks_per_update: 1,
            ticks_per_second: 0.0,
            last_tick_count: 0,
            last_sample: Instant::now(),
            last_error: None,
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ryzr matrix editor".to_owned(),
                resolution: WindowResolution::new(1280, 800),
                present_mode: PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.035, 0.039, 0.044)))
        .insert_resource(UiState::default())
        .insert_non_send_resource(
            EditorRuntime::new(Board::demo()).expect("demo board must compile"),
        )
        .add_systems(Startup, setup_ui)
        .add_systems(
            Update,
            (
                handle_palette_buttons,
                handle_cell_buttons,
                handle_control_buttons,
                run_runtime,
                update_cell_views,
                update_palette_views,
                update_control_labels,
                update_metrics,
            ),
        )
        .run();
}

#[allow(clippy::needless_pass_by_value)]
fn setup_ui(mut commands: Commands, runtime: NonSend<EditorRuntime>) {
    commands.spawn(Camera2d);
    commands
        .spawn((
            Node {
                width: percent(100),
                height: percent(100),
                padding: UiRect::all(px(12)),
                column_gap: px(12),
                align_items: AlignItems::Stretch,
                ..default()
            },
            BackgroundColor(Color::srgb(0.035, 0.039, 0.044)),
        ))
        .with_children(|root| {
            spawn_palette(root);
            spawn_board(root, runtime.board());
            spawn_inspector(root);
        });
}

fn spawn_palette(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                width: px(LEFT_PANEL_WIDTH),
                height: percent(100),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(px(12)),
                row_gap: px(8),
                ..default()
            },
            BackgroundColor(panel_color()),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("ryzr board"),
                TextFont { font_size: 22.0, ..default() },
                TextColor(Color::srgb(0.92, 0.94, 0.91)),
            ));
            panel.spawn((
                Text::new("components"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::srgb(0.58, 0.64, 0.68)),
            ));
            for &kind in CellKind::palette() {
                spawn_palette_button(panel, kind);
            }
        });
}

fn spawn_board(parent: &mut ChildSpawnerCommands, board: &Board) {
    parent
        .spawn((
            Node {
                flex_grow: 1.0,
                height: percent(100),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::NONE),
        ))
        .with_children(|center| {
            center
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        padding: UiRect::all(px(10)),
                        row_gap: px(CELL_GAP),
                        border: UiRect::all(px(1)),
                        ..default()
                    },
                    BorderColor::all(Color::srgba(0.46, 0.62, 0.66, 0.85)),
                    BackgroundColor(Color::srgb(0.055, 0.066, 0.074)),
                ))
                .with_children(|grid| {
                    for y in 0..board.height() {
                        grid.spawn((
                            Node {
                                width: auto(),
                                height: px(CELL_SIZE),
                                column_gap: px(CELL_GAP),
                                ..default()
                            },
                            BackgroundColor(Color::NONE),
                        ))
                        .with_children(|row| {
                            for x in 0..board.width() {
                                spawn_cell(row, CellPos::new(x, y));
                            }
                        });
                    }
                });
        });
}

fn spawn_inspector(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                width: px(RIGHT_PANEL_WIDTH),
                height: percent(100),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(px(12)),
                row_gap: px(10),
                ..default()
            },
            BackgroundColor(panel_color()),
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("simulation"),
                TextFont { font_size: 22.0, ..default() },
                TextColor(Color::srgb(0.92, 0.94, 0.91)),
            ));
            spawn_control_row(
                panel,
                &[
                    (ControlKind::Step, "Step"),
                    (ControlKind::RunPause, "Run"),
                    (ControlKind::Reset, "Reset"),
                ],
            );
            spawn_control_row(
                panel,
                &[
                    (ControlKind::Slower, "-"),
                    (ControlKind::Faster, "+"),
                    (ControlKind::ToggleInput, "Toggle"),
                ],
            );
            spawn_control_row(panel, &[(ControlKind::Demo, "Demo"), (ControlKind::Clear, "Clear")]);
            spawn_metric(panel, MetricKind::Status, 15.0);
            spawn_metric(panel, MetricKind::Circuit, 15.0);
            spawn_metric(panel, MetricKind::Selected, 15.0);
            spawn_metric(panel, MetricKind::Sources, 14.0);
            spawn_metric(panel, MetricKind::Leds, 14.0);
            spawn_metric(panel, MetricKind::Vcb, 14.0);
            spawn_metric(panel, MetricKind::Error, 13.0);
        });
}

fn spawn_palette_button(parent: &mut ChildSpawnerCommands, kind: CellKind) {
    parent
        .spawn((
            Button,
            PaletteButton(kind),
            Node {
                width: percent(100),
                height: px(34),
                padding: UiRect::horizontal(px(10)),
                justify_content: JustifyContent::SpaceBetween,
                align_items: AlignItems::Center,
                border: UiRect::all(px(1)),
                ..default()
            },
            BorderColor::all(Color::srgba(0.34, 0.39, 0.42, 0.9)),
            BackgroundColor(button_color(false)),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(kind.name()),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::srgb(0.86, 0.89, 0.88)),
            ));
            button.spawn((
                Text::new(kind.label()),
                TextFont { font_size: 11.0, ..default() },
                TextColor(kind_text_color(kind, false)),
            ));
        });
}

fn spawn_cell(parent: &mut ChildSpawnerCommands, pos: CellPos) {
    parent
        .spawn((
            Button,
            CellButton { pos },
            Node {
                width: px(CELL_SIZE),
                height: px(CELL_SIZE),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                border: UiRect::all(px(1)),
                ..default()
            },
            BorderColor::all(Color::srgba(0.16, 0.20, 0.23, 0.9)),
            BackgroundColor(cell_color(CellKind::Empty, false, false)),
        ))
        .with_children(|cell| {
            cell.spawn((
                Text::new(""),
                TextFont { font_size: 9.5, ..default() },
                TextColor(Color::srgb(0.72, 0.76, 0.75)),
                CellLabel { pos },
            ));
        });
}

fn spawn_control_row(parent: &mut ChildSpawnerCommands, controls: &[(ControlKind, &str)]) {
    parent
        .spawn((
            Node { width: percent(100), height: px(36), column_gap: px(8), ..default() },
            BackgroundColor(Color::NONE),
        ))
        .with_children(|row| {
            for &(kind, label) in controls {
                spawn_control_button(row, kind, label);
            }
        });
}

fn spawn_control_button(parent: &mut ChildSpawnerCommands, kind: ControlKind, label: &str) {
    parent
        .spawn((
            Button,
            ControlButton(kind),
            Node {
                flex_grow: 1.0,
                min_width: px(56),
                height: px(34),
                padding: UiRect::horizontal(px(10)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                border: UiRect::all(px(1)),
                ..default()
            },
            BorderColor::all(Color::srgba(0.42, 0.49, 0.51, 0.9)),
            BackgroundColor(button_color(false)),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(label),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::srgb(0.90, 0.93, 0.91)),
                ControlButtonLabel(kind),
            ));
        });
}

fn spawn_metric(parent: &mut ChildSpawnerCommands, kind: MetricKind, size: f32) {
    parent.spawn((
        Text::new(""),
        TextFont { font_size: size, ..default() },
        TextColor(Color::srgb(0.82, 0.86, 0.84)),
        Node { width: percent(100), ..default() },
        MetricText(kind),
    ));
}

fn handle_palette_buttons(
    mut state: ResMut<UiState>,
    interactions: Query<(&Interaction, &PaletteButton), Changed<Interaction>>,
) {
    for (interaction, button) in &interactions {
        if *interaction == Interaction::Pressed {
            state.selected_tool = button.0;
        }
    }
}

fn handle_cell_buttons(
    mut state: ResMut<UiState>,
    mut runtime: NonSendMut<EditorRuntime>,
    interactions: Query<(&Interaction, &CellButton), Changed<Interaction>>,
) {
    for (interaction, button) in &interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }

        state.selected_cell = Some(button.pos);
        let current = runtime
            .board()
            .cell(button.pos.x, button.pos.y)
            .map_or(CellKind::Empty, |cell| cell.kind);

        let result = if current == state.selected_tool && current.is_source() {
            runtime.toggle_input(button.pos.x, button.pos.y)
        } else {
            runtime.set_kind(button.pos.x, button.pos.y, state.selected_tool)
        };
        remember_result(&mut state, result);
    }
}

fn handle_control_buttons(
    mut state: ResMut<UiState>,
    mut runtime: NonSendMut<EditorRuntime>,
    interactions: Query<(&Interaction, &ControlButton), Changed<Interaction>>,
) {
    for (interaction, button) in &interactions {
        if *interaction != Interaction::Pressed {
            continue;
        }

        match button.0 {
            ControlKind::Step => runtime.step(),
            ControlKind::RunPause => state.running = !state.running,
            ControlKind::Slower => state.ticks_per_update = (state.ticks_per_update / 2).max(1),
            ControlKind::Faster => {
                state.ticks_per_update = (state.ticks_per_update * 2).min(MAX_TICKS_PER_UPDATE);
            }
            ControlKind::Reset => remember_result(&mut state, runtime.reset()),
            ControlKind::ToggleInput => {
                let result =
                    state.selected_cell.ok_or_else(|| "no selected source".to_owned()).and_then(
                        |pos| runtime.toggle_input(pos.x, pos.y).map_err(|err| err.to_string()),
                    );
                remember_text_result(&mut state, result);
            }
            ControlKind::Demo => {
                state.selected_cell = None;
                remember_result(&mut state, runtime.load_demo());
            }
            ControlKind::Clear => {
                state.selected_cell = None;
                remember_result(&mut state, runtime.clear());
            }
        }
    }
}

fn run_runtime(mut state: ResMut<UiState>, mut runtime: NonSendMut<EditorRuntime>) {
    if state.running {
        runtime.run(state.ticks_per_update);
    }

    let elapsed = state.last_sample.elapsed();
    if elapsed.as_millis() >= 250 {
        let tick_count = runtime.tick_count();
        let delta = tick_count.saturating_sub(state.last_tick_count);
        state.ticks_per_second = delta as f64 / elapsed.as_secs_f64();
        state.last_tick_count = tick_count;
        state.last_sample = Instant::now();
    }
}

#[allow(clippy::needless_pass_by_value)]
fn update_cell_views(
    state: Res<UiState>,
    runtime: NonSend<EditorRuntime>,
    mut cells: Query<(&CellButton, &mut BackgroundColor, &mut BorderColor)>,
    mut labels: Query<(&CellLabel, &mut Text, &mut TextColor)>,
) {
    for (button, mut background, mut border) in &mut cells {
        let kind = runtime
            .board()
            .cell(button.pos.x, button.pos.y)
            .map_or(CellKind::Empty, |cell| cell.kind);
        let active = runtime.cell_value(button.pos.x, button.pos.y);
        let selected = state.selected_cell == Some(button.pos);
        *background = BackgroundColor(cell_color(kind, active, selected));
        *border = BorderColor::all(if selected {
            Color::srgb(0.97, 0.73, 0.34)
        } else {
            Color::srgba(0.16, 0.20, 0.23, 0.9)
        });
    }

    for (label, mut text, mut color) in &mut labels {
        let Some(cell) = runtime.board().cell(label.pos.x, label.pos.y) else {
            continue;
        };
        **text = cell_text(cell.kind, runtime.cell_value(label.pos.x, label.pos.y));
        *color =
            TextColor(kind_text_color(cell.kind, runtime.cell_value(label.pos.x, label.pos.y)));
    }
}

#[allow(clippy::needless_pass_by_value)]
fn update_palette_views(
    state: Res<UiState>,
    mut query: Query<(&PaletteButton, &mut BackgroundColor, &mut BorderColor)>,
) {
    for (button, mut background, mut border) in &mut query {
        let selected = button.0 == state.selected_tool;
        *background = BackgroundColor(button_color(selected));
        *border = BorderColor::all(if selected {
            Color::srgb(0.92, 0.68, 0.32)
        } else {
            Color::srgba(0.34, 0.39, 0.42, 0.9)
        });
    }
}

#[allow(clippy::needless_pass_by_value)]
fn update_control_labels(state: Res<UiState>, mut query: Query<(&ControlButtonLabel, &mut Text)>) {
    for (label, mut text) in &mut query {
        **text = control_label(label.0, &state);
    }
}

#[allow(clippy::needless_pass_by_value)]
fn update_metrics(
    state: Res<UiState>,
    runtime: NonSend<EditorRuntime>,
    mut query: Query<(&MetricText, &mut Text, &mut TextColor)>,
) {
    for (metric, mut text, mut color) in &mut query {
        **text = metric_text(metric.0, &state, &runtime);
        *color = TextColor(if metric.0 == MetricKind::Error && state.last_error.is_some() {
            Color::srgb(0.96, 0.50, 0.42)
        } else {
            Color::srgb(0.82, 0.86, 0.84)
        });
    }
}

fn metric_text(kind: MetricKind, state: &UiState, runtime: &EditorRuntime) -> String {
    match kind {
        MetricKind::Status => {
            let mode = if state.running { "running" } else { "paused" };
            format!(
                "{mode}\nselected: {}\nbudget: {} tick/update",
                state.selected_tool.name(),
                state.ticks_per_update
            )
        }
        MetricKind::Circuit => {
            let stats = runtime.circuit_stats();
            format!(
                "engine: {}\nticks: {}\nspeed: {:>7.0} tick/s\ngates: {}  regs: {}\nsignals: {}",
                runtime.engine_name(),
                runtime.tick_count(),
                state.ticks_per_second,
                stats.gates,
                stats.registers,
                stats.signals
            )
        }
        MetricKind::Selected => selected_text(state, runtime),
        MetricKind::Sources => source_text(runtime),
        MetricKind::Leds => led_text(runtime),
        MetricKind::Vcb => format!(
            "ViPeR nominal clock: {VIPER_CLOCK_TICKS_PER_CPU_CYCLE} VCB ticks/cycle\nnormalized VCB ticks: {}",
            runtime.tick_count() * VIPER_CLOCK_TICKS_PER_CPU_CYCLE
        ),
        MetricKind::Error => state.last_error.clone().unwrap_or_default(),
    }
}

fn selected_text(state: &UiState, runtime: &EditorRuntime) -> String {
    let Some(pos) = state.selected_cell else {
        return "cell: none".to_owned();
    };
    let Some(cell) = runtime.board().cell(pos.x, pos.y) else {
        return "cell: out of range".to_owned();
    };
    let value = u8::from(runtime.cell_value(pos.x, pos.y));
    if let Some(input) = runtime.cell_input_value(pos.x, pos.y) {
        format!(
            "cell: {},{}\nkind: {}\nsource: {}\nvalue: {value}",
            pos.x,
            pos.y,
            cell.kind.name(),
            u8::from(input)
        )
    } else {
        format!("cell: {},{}\nkind: {}\nvalue: {value}", pos.x, pos.y, cell.kind.name())
    }
}

fn source_text(runtime: &EditorRuntime) -> String {
    let mut rows = Vec::new();
    for y in 0..runtime.board().height() {
        for x in 0..runtime.board().width() {
            let Some(cell) = runtime.board().cell(x, y) else {
                continue;
            };
            if cell.kind.is_source() {
                rows.push(format!(
                    "{} {},{}={}",
                    cell.kind.label(),
                    x,
                    y,
                    u8::from(cell.input_value)
                ));
            }
        }
    }
    if rows.is_empty() {
        "sources: none".to_owned()
    } else {
        format!("sources:\n{}", rows.join("\n"))
    }
}

fn led_text(runtime: &EditorRuntime) -> String {
    let mut rows = Vec::new();
    for y in 0..runtime.board().height() {
        for x in 0..runtime.board().width() {
            let Some(cell) = runtime.board().cell(x, y) else {
                continue;
            };
            if cell.kind == CellKind::Led {
                rows.push(format!("LED {},{}={}", x, y, u8::from(runtime.cell_value(x, y))));
            }
        }
    }
    if rows.is_empty() { "LEDs: none".to_owned() } else { format!("LEDs:\n{}", rows.join("\n")) }
}

fn cell_text(kind: CellKind, active: bool) -> String {
    if kind == CellKind::Empty {
        String::new()
    } else {
        format!("{}\n{}", kind.label(), u8::from(active))
    }
}

fn control_label(kind: ControlKind, state: &UiState) -> String {
    match kind {
        ControlKind::RunPause => {
            if state.running {
                "Pause".to_owned()
            } else {
                "Run".to_owned()
            }
        }
        ControlKind::Step => "Step".to_owned(),
        ControlKind::Slower => "-".to_owned(),
        ControlKind::Faster => "+".to_owned(),
        ControlKind::Reset => "Reset".to_owned(),
        ControlKind::ToggleInput => "Toggle".to_owned(),
        ControlKind::Demo => "Demo".to_owned(),
        ControlKind::Clear => "Clear".to_owned(),
    }
}

fn remember_result(state: &mut UiState, result: Result<(), impl ToString>) {
    remember_text_result(state, result.map_err(|err| err.to_string()));
}

fn remember_text_result(state: &mut UiState, result: Result<(), String>) {
    state.last_error = result.err();
}

fn cell_color(kind: CellKind, active: bool, selected: bool) -> Color {
    let mut color = match kind {
        CellKind::Empty => Color::srgb(0.070, 0.083, 0.092),
        CellKind::Input => Color::srgb(0.12, 0.28, 0.22),
        CellKind::Clock => Color::srgb(0.32, 0.24, 0.10),
        CellKind::Wire => Color::srgb(0.10, 0.21, 0.33),
        CellKind::Not => Color::srgb(0.28, 0.20, 0.32),
        CellKind::And => Color::srgb(0.18, 0.26, 0.36),
        CellKind::Or => Color::srgb(0.18, 0.30, 0.24),
        CellKind::Xor => Color::srgb(0.30, 0.22, 0.18),
        CellKind::Nand => Color::srgb(0.32, 0.18, 0.18),
        CellKind::Register => Color::srgb(0.20, 0.22, 0.35),
        CellKind::Led => Color::srgb(0.18, 0.18, 0.20),
    };
    if active {
        color = match kind {
            CellKind::Led => Color::srgb(0.98, 0.72, 0.22),
            CellKind::Clock => Color::srgb(0.85, 0.53, 0.18),
            CellKind::Input => Color::srgb(0.24, 0.58, 0.38),
            _ => Color::srgb(0.25, 0.52, 0.62),
        };
    }
    if selected {
        color = color.mix(&Color::srgb(0.96, 0.68, 0.25), 0.22);
    }
    color
}

fn kind_text_color(kind: CellKind, active: bool) -> Color {
    if active {
        return Color::srgb(0.98, 0.96, 0.82);
    }
    match kind {
        CellKind::Empty => Color::srgb(0.50, 0.55, 0.55),
        CellKind::Led => Color::srgb(0.96, 0.70, 0.34),
        CellKind::Input | CellKind::Clock => Color::srgb(0.78, 0.92, 0.80),
        _ => Color::srgb(0.78, 0.84, 0.86),
    }
}

fn button_color(selected: bool) -> Color {
    if selected { Color::srgb(0.26, 0.23, 0.16) } else { Color::srgb(0.10, 0.12, 0.13) }
}

fn panel_color() -> Color {
    Color::srgba(0.055, 0.064, 0.071, 0.96)
}
