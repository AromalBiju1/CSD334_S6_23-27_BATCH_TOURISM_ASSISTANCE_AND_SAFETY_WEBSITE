import React from "react";

function BackgroundBlur() {
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

