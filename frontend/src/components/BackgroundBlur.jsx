import React from "react";
import { useTheme } from "../context/ThemeContext";

function BackgroundBlur() {
  const { theme } = useTheme();

  if (theme === 'light') {
    return null; // No blur effect in light mode
  }

  return (
    <div
      className="
        fixed
        top-1/2 left-1/2
        -translate-x-1/2 -translate-y-1/2
        w-[600px] h-[600px]
        bg-[#01bc82]
        blur-[250px]
        opacity-45
        rounded-full
        pointer-events-none
      "
    />
  );
}

export default BackgroundBlur;
