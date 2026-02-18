%% DC_MICROGRID_VISUALIZE.m
% DC Microgrid Circuit Schematic Visualization with Fault Highlighting
% 
% This MATLAB script renders a proper electrical circuit schematic with:
%   - Bus bars as horizontal/vertical lines
%   - Cables connecting buses
%   - Generator and Load symbols (standard electrical notation)
%   - Real-time fault location highlighting
%
% Usage:
%   result = DC_Microgrid_Visualize(circuit_data)
%   result = DC_Microgrid_Visualize(circuit_data, fault_data)
%
% Inputs:
%   circuit_data - struct with fields: buses, lines, generators, loads
%   fault_data   - optional struct with fields: active, location, type, severity
%
% Output:
%   result - struct with figure handle and update function
%
% Author: DC Microgrid Protection Platform
% Version: 2.0

function result = DC_Microgrid_Visualize(circuit_data, fault_data)

    %% Initialize
    if nargin < 2
        fault_data = struct('active', false, 'location', '', 'type', '', 'severity', 0);
    end
    
    % Color scheme
    colors = struct(...
        'background', [0.04 0.04 0.12], ...
        'bus_normal', [0.1 0.6 0.4], ...
        'bus_fault', [0.9 0.2 0.3], ...
        'line_normal', [0.3 0.5 0.7], ...
        'line_fault', [1.0 0.3 0.2], ...
        'generator', [0.2 0.7 0.9], ...
        'load', [0.9 0.6 0.2], ...
        'text', [0.9 0.9 0.95], ...
        'grid', [0.2 0.2 0.3] ...
    );
    
    %% Create Figure
    fig = figure('Name', 'DC Microgrid Digital Twin', ...
                 'NumberTitle', 'off', ...
                 'Color', colors.background, ...
                 'Position', [100, 100, 1200, 800], ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none');
    
    hold on;
    axis equal;
    axis off;
    
    %% Parse Circuit Data
    buses = circuit_data.buses;
    lines = circuit_data.lines;
    generators = circuit_data.generators;
    loads = circuit_data.loads;
    
    n_buses = length(buses);
    n_lines = length(lines);
    
    %% Build Position Map
    % Scale and center the circuit for display
    x_coords = [buses.x];
    y_coords = [buses.y];
    
    % Normalize to [0.1, 0.9] range
    x_range = max(x_coords) - min(x_coords);
    y_range = max(y_coords) - min(y_coords);
    
    if x_range == 0, x_range = 1; end
    if y_range == 0, y_range = 1; end
    
    pos_map = containers.Map();
    for i = 1:n_buses
        bid = buses(i).id;
        pos_map(num2str(bid)) = struct(...
            'x', 0.1 + 0.8 * (buses(i).x - min(x_coords)) / x_range, ...
            'y', 0.1 + 0.8 * (buses(i).y - min(y_coords)) / y_range, ...
            'name', buses(i).name, ...
            'type', buses(i).type ...
        );
    end
    
    %% Determine Fault Location
    fault_bus_id = '';
    fault_line_id = '';
    
    if fault_data.active
        % Check if fault is on a bus
        for i = 1:n_buses
            if strcmp(num2str(buses(i).id), fault_data.location)
                fault_bus_id = num2str(buses(i).id);
                break;
            end
        end
        
        % Check if fault is on a line
        for i = 1:n_lines
            if strcmp(num2str(lines(i).id), fault_data.location)
                fault_line_id = num2str(lines(i).id);
                break;
            end
        end
    end
    
    %% Draw Grid Background
    for gx = 0:0.1:1
        plot([gx, gx], [0, 1], 'Color', colors.grid, 'LineWidth', 0.5);
    end
    for gy = 0:0.1:1
        plot([0, 1], [gy, gy], 'Color', colors.grid, 'LineWidth', 0.5);
    end
    
    %% Draw Lines (Cables)
    line_handles = [];
    for i = 1:n_lines
        from_id = num2str(lines(i).from_bus);
        to_id = num2str(lines(i).to_bus);
        
        if isKey(pos_map, from_id) && isKey(pos_map, to_id)
            from_pos = pos_map(from_id);
            to_pos = pos_map(to_id);
            
            % Determine if this line has fault
            is_fault_line = strcmp(num2str(lines(i).id), fault_line_id);
            
            if is_fault_line
                line_color = colors.line_fault;
                line_width = 4;
            else
                line_color = colors.line_normal;
                line_width = 2;
            end
            
            % Draw cable with orthogonal routing
            mid_x = (from_pos.x + to_pos.x) / 2;
            
            % Draw as orthogonal path (L-shaped or Z-shaped)
            if abs(from_pos.x - to_pos.x) > 0.15
                % Z-shaped routing
                plot([from_pos.x, mid_x], [from_pos.y, from_pos.y], ...
                     'Color', line_color, 'LineWidth', line_width);
                plot([mid_x, mid_x], [from_pos.y, to_pos.y], ...
                     'Color', line_color, 'LineWidth', line_width);
                plot([mid_x, to_pos.x], [to_pos.y, to_pos.y], ...
                     'Color', line_color, 'LineWidth', line_width);
            else
                % Direct line for close buses
                plot([from_pos.x, to_pos.x], [from_pos.y, to_pos.y], ...
                     'Color', line_color, 'LineWidth', line_width);
            end
            
            % Line label
            mid_y = (from_pos.y + to_pos.y) / 2;
            text(mid_x + 0.02, mid_y + 0.02, sprintf('L%d', lines(i).id), ...
                 'Color', colors.text * 0.7, 'FontSize', 8, 'FontName', 'Consolas');
            
            % Fault marker on line
            if is_fault_line
                % Draw fault symbol (lightning bolt)
                draw_fault_symbol(mid_x, mid_y, fault_data.type, colors);
                
                % Pulsing animation effect (draw multiple circles)
                for r = 0.03:0.01:0.06
                    rectangle('Position', [mid_x-r, mid_y-r, 2*r, 2*r], ...
                              'Curvature', 1, 'EdgeColor', colors.line_fault, ...
                              'LineWidth', 2, 'LineStyle', '--');
                end
            end
        end
    end
    
    %% Draw Buses (Bus Bars)
    bus_handles = [];
    for i = 1:n_buses
        bid = num2str(buses(i).id);
        
        if isKey(pos_map, bid)
            pos = pos_map(bid);
            
            % Determine if this bus has fault
            is_fault_bus = strcmp(bid, fault_bus_id);
            
            if is_fault_bus
                bus_color = colors.bus_fault;
                bus_size = 0.04;
            else
                bus_color = colors.bus_normal;
                bus_size = 0.025;
            end
            
            % Draw bus bar (horizontal rectangle for DC bus)
            bus_width = 0.08;
            bus_height = 0.015;
            
            % Bus bar
            fill([pos.x-bus_width/2, pos.x+bus_width/2, pos.x+bus_width/2, pos.x-bus_width/2], ...
                 [pos.y-bus_height/2, pos.y-bus_height/2, pos.y+bus_height/2, pos.y+bus_height/2], ...
                 bus_color, 'EdgeColor', 'white', 'LineWidth', 1.5);
            
            % Bus label
            text(pos.x, pos.y + 0.04, pos.name, ...
                 'Color', colors.text, 'FontSize', 10, 'FontWeight', 'bold', ...
                 'HorizontalAlignment', 'center', 'FontName', 'Arial');
            
            % Bus type indicator
            type_text = '';
            if strcmp(pos.type, 'Slack')
                type_text = '⬢ GRID';
            elseif strcmp(pos.type, 'PV')
                type_text = '◉ PV';
            else
                type_text = '○ LOAD';
            end
            text(pos.x, pos.y - 0.035, type_text, ...
                 'Color', colors.text * 0.8, 'FontSize', 8, ...
                 'HorizontalAlignment', 'center', 'FontName', 'Arial');
            
            % Voltage display
            if isfield(buses(i), 'voltage')
                v_text = sprintf('%.1f V', buses(i).voltage * 1000);
                text(pos.x, pos.y - 0.06, v_text, ...
                     'Color', colors.text * 0.7, 'FontSize', 9, ...
                     'HorizontalAlignment', 'center', 'FontName', 'Consolas');
            end
            
            % Fault indicator on bus
            if is_fault_bus
                % Draw pulsing fault circles
                for r = 0.04:0.015:0.08
                    rectangle('Position', [pos.x-r, pos.y-r, 2*r, 2*r], ...
                              'Curvature', 1, 'EdgeColor', colors.bus_fault, ...
                              'LineWidth', 2, 'LineStyle', '--');
                end
                
                % Draw fault symbol
                draw_fault_symbol(pos.x + 0.06, pos.y + 0.03, fault_data.type, colors);
                
                % Fault type label
                fault_label = sprintf('FAULT: %s', fault_data.type);
                text(pos.x, pos.y + 0.09, fault_label, ...
                     'Color', colors.bus_fault, 'FontSize', 12, ...
                     'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                     'BackgroundColor', [0.2 0.05 0.05], 'EdgeColor', colors.bus_fault);
            end
        end
    end
    
    %% Draw Generators
    for i = 1:length(generators)
        gen_bus_id = num2str(generators(i).bus_id);
        
        if isKey(pos_map, gen_bus_id)
            pos = pos_map(gen_bus_id);
            
            % Draw generator symbol (circle with G)
            gen_x = pos.x - 0.07;
            gen_y = pos.y + 0.06;
            
            % Generator circle
            th = linspace(0, 2*pi, 50);
            r = 0.025;
            fill(gen_x + r*cos(th), gen_y + r*sin(th), colors.generator, ...
                 'EdgeColor', 'white', 'LineWidth', 1.5);
            
            % Generator label
            text(gen_x, gen_y, 'G', 'Color', [0 0 0], 'FontSize', 10, ...
                 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
            
            % Power output
            if isfield(generators(i), 'p_mw')
                p_text = sprintf('%.1f MW', generators(i).p_mw);
                text(gen_x, gen_y - 0.04, p_text, 'Color', colors.text * 0.8, ...
                     'FontSize', 8, 'HorizontalAlignment', 'center');
            end
            
            % Connection line to bus
            plot([gen_x, pos.x-bus_width/2], [gen_y-0.025, pos.y], ...
                 'Color', colors.generator, 'LineWidth', 1.5, 'LineStyle', '-.');
        end
    end
    
    %% Draw Loads
    for i = 1:length(loads)
        load_bus_id = num2str(loads(i).bus_id);
        
        if isKey(pos_map, load_bus_id)
            pos = pos_map(load_bus_id);
            
            % Draw load symbol (arrow pointing down)
            load_x = pos.x + 0.07;
            load_y = pos.y - 0.06;
            
            % Load triangle/arrow
            load_size = 0.03;
            fill([load_x, load_x-load_size, load_x+load_size], ...
                 [load_y, load_y+load_size*1.5, load_y+load_size*1.5], ...
                 colors.load, 'EdgeColor', 'white', 'LineWidth', 1.5);
            
            % Load label
            text(load_x, load_y + 0.05, sprintf('L%d', loads(i).id), ...
                 'Color', colors.text * 0.8, 'FontSize', 8, ...
                 'HorizontalAlignment', 'center');
            
            % Power consumption
            if isfield(loads(i), 'p_mw')
                p_text = sprintf('%.1f MW', loads(i).p_mw);
                text(load_x, load_y - 0.03, p_text, 'Color', colors.text * 0.8, ...
                     'FontSize', 8, 'HorizontalAlignment', 'center');
            end
            
            % Connection line to bus
            plot([load_x, pos.x+bus_width/2], [load_y+load_size*1.5, pos.y], ...
                 'Color', colors.load, 'LineWidth', 1.5, 'LineStyle', '-.');
        end
    end
    
    %% Draw Title and Legend
    title('DC Microgrid Circuit Schematic', 'Color', colors.text, ...
          'FontSize', 16, 'FontWeight', 'bold');
    
    % Status indicator
    if fault_data.active
        status_color = colors.bus_fault;
        status_text = sprintf('⚠ FAULT ACTIVE: %s at %s (Severity: %.0f%%)', ...
                              fault_data.type, fault_data.location, fault_data.severity * 100);
    else
        status_color = colors.bus_normal;
        status_text = '✓ System Normal';
    end
    
    annotation('textbox', [0.15, 0.92, 0.7, 0.05], 'String', status_text, ...
               'Color', status_color, 'FontSize', 14, 'FontWeight', 'bold', ...
               'HorizontalAlignment', 'center', 'EdgeColor', 'none', ...
               'BackgroundColor', colors.background * 0.5);
    
    % Legend
    legend_x = 0.85;
    legend_y = 0.25;
    
    % Legend box
    fill([legend_x-0.02, legend_x+0.13, legend_x+0.13, legend_x-0.02], ...
         [legend_y-0.02, legend_y-0.02, legend_y+0.18, legend_y+0.18], ...
         colors.background * 1.5, 'EdgeColor', colors.grid);
    
    text(legend_x+0.055, legend_y+0.14, 'Legend', 'Color', colors.text, ...
         'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    
    % Legend items
    plot([legend_x, legend_x+0.03], [legend_y+0.10, legend_y+0.10], ...
         'Color', colors.line_normal, 'LineWidth', 2);
    text(legend_x+0.04, legend_y+0.10, 'Cable', 'Color', colors.text, 'FontSize', 8);
    
    fill([legend_x, legend_x+0.02, legend_x+0.02, legend_x], ...
         [legend_y+0.05, legend_y+0.05, legend_y+0.07, legend_y+0.07], ...
         colors.bus_normal);
    text(legend_x+0.04, legend_y+0.06, 'Bus', 'Color', colors.text, 'FontSize', 8);
    
    fill([legend_x, legend_x+0.02, legend_x+0.01], ...
         [legend_y, legend_y+0.02, legend_y+0.02], colors.generator);
    text(legend_x+0.04, legend_y+0.01, 'Generator', 'Color', colors.text, 'FontSize', 8);
    
    %% Set axis limits
    xlim([-0.05, 1.05]);
    ylim([-0.05, 1.05]);
    
    %% Create result structure
    result.figure = fig;
    result.colors = colors;
    result.pos_map = pos_map;
    
    % Update function handle
    result.update = @(new_fault) update_fault_display(fig, new_fault, pos_map, colors);
    
    hold off;
end

%% Helper: Draw Fault Symbol
function draw_fault_symbol(x, y, fault_type, colors)
    % Draw lightning bolt or fault indicator
    
    switch fault_type
        case 'LINE_TO_LINE'
            % Lightning bolt
            bolt_x = [0, 0.015, 0.005, 0.02, -0.005, 0.005, -0.01];
            bolt_y = [0.03, 0.01, 0.01, -0.015, 0, 0, 0.02];
            fill(x + bolt_x, y + bolt_y, colors.bus_fault, 'EdgeColor', 'white');
            
        case 'LINE_TO_GROUND'
            % Ground symbol with arc
            th = linspace(pi/4, 3*pi/4, 20);
            r = 0.02;
            plot(x + r*cos(th), y + r*sin(th), 'Color', colors.bus_fault, 'LineWidth', 2);
            plot([x-0.015, x+0.015], [y-0.01, y-0.01], 'Color', colors.bus_fault, 'LineWidth', 2);
            
        case 'ARC_FAULT'
            % Arc symbol (zigzag)
            arc_x = linspace(-0.015, 0.015, 20);
            arc_y = 0.01 * sin(arc_x * 200) + 0.01;
            plot(x + arc_x, y + arc_y, 'Color', colors.bus_fault, 'LineWidth', 2);
            
        otherwise
            % Default fault indicator (exclamation in circle)
            th = linspace(0, 2*pi, 50);
            r = 0.02;
            plot(x + r*cos(th), y + r*sin(th), 'Color', colors.bus_fault, 'LineWidth', 2);
            text(x, y, '!', 'Color', colors.bus_fault, 'FontSize', 12, ...
                 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
end

%% Helper: Update Fault Display
function update_fault_display(fig, new_fault, pos_map, colors)
    % This function would update an existing figure with new fault data
    % Implementation for real-time updates
    
    figure(fig);
    
    % Clear previous fault indicators (would need to track handles)
    % For simplicity, this is a placeholder for the update mechanism
    
    if new_fault.active
        % Add new fault indicators
        fprintf('Fault updated: %s at %s\n', new_fault.type, new_fault.location);
    end
    
    drawnow;
end
