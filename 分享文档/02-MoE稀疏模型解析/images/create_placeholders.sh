#!/bin/bash
# 使用ImageMagick或sips创建占位图片

create_placeholder() {
    local name=$1
    local text=$2
    
    if command -v convert &> /dev/null; then
        convert -size 800x600 xc:white -pointsize 40 -gravity center -annotate +0+0 "$text" "${name}.png"
        echo "Created ${name}.png using ImageMagick"
    elif command -v sips &> /dev/null; then
        # Mac系统使用sips
        sips -s format png --setProperty formatOptions 100 /System/Library/CoreServices/DefaultDesktop.heic --out "${name}.png" 2>/dev/null || \
        echo "800 600" | awk '{for(i=0;i<$1*$2;i++)print "255 255 255"}' | \
        convert -size 800x600 -depth 8 rgb:- -pointsize 40 -gravity center -annotate +0+0 "$text" "${name}.png" 2>/dev/null || \
        echo "Cannot create ${name}.png - please install ImageMagick or use Python PIL"
    else
        echo "Cannot create ${name}.png - no image tools available"
    fi
}

create_placeholder "jacobs1991" "Jacobs 1991\nMoE Architecture\n(请替换为实际图片)"
create_placeholder "shazeer2017" "Shazeer 2017\nSparse MoE Architecture\n(请替换为实际图片)"
create_placeholder "switch2021" "Switch Transformer\nArchitecture\n(请替换为实际图片)"
create_placeholder "mixtral2024" "Mixtral 8x7B\nArchitecture\n(请替换为实际图片)"
create_placeholder "moe_transformer_layer" "MoE Transformer Layer\nArchitecture\n(请替换为实际图片)"
