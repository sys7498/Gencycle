import { Component, ElementRef, EventEmitter, Input, Output, Renderer2, ViewChild } from '@angular/core';

@Component({
  selector: 'app-select-window',
  templateUrl: './select-window.component.html',
  styleUrl: './select-window.component.scss'
})
export class SelectWindowComponent {
  @Input() width: string = '500px'; // 창의 기본 너비
  @Input() minHeight: string = '300px'; // 최소 높이
  @Input() check: boolean = true; // 보조 버튼 내용
  @Input() title: string = 'Title';
  @Input() confirmButtonLabel: string = ''; // 기본 버튼 내용
  @Input() cancelButtonLabel: string = ''; // 보조 버튼 내용
  @Output() onClickCancelButton = new EventEmitter<void>(); // 주 버튼 클릭 이벤트
  @Output() onClickConfirmButton = new EventEmitter<void>(); // 보조 버튼 클릭 이벤트
  @ViewChild('windowContainer', { static: true }) windowContainer!: ElementRef;

  constructor(private renderer: Renderer2) {}

  // 마우스 움직임에 따라 회전값 계산
  onMouseMove(event: MouseEvent) {
    const { offsetWidth, offsetHeight } = this.windowContainer.nativeElement;
    const x = (event.offsetX / offsetWidth - 0.5) * 10; // X축 회전 (-5deg ~ 5deg)
    const y = (event.offsetY / offsetHeight - 0.5) * -10; // Y축 회전 (-5deg ~ 5deg)

    // 스타일에 3D 회전 적용
    //this.renderer.setStyle(
    //  this.windowContainer.nativeElement,
    //  'transform',
    //  `perspective(1000px) rotateX(${y}deg) rotateY(${x}deg)`
    //);

    // 빛 반사 위치 조정
    //const highlight = this.windowContainer.nativeElement.querySelector('.light-highlight');
    //this.renderer.setStyle(highlight, 'background', `radial-gradient(circle at ${event.offsetX}px ${event.offsetY}px, rgba(255, 255, 255, 0.3), transparent)`);
  }

  // 마우스가 떠날 때 초기화
  onMouseLeave() {
    //this.renderer.setStyle(this.windowContainer.nativeElement, 'transform', 'perspective(1000px) rotateX(0deg) rotateY(0deg)');
  }

  onClickConfirm(){
    this.onClickConfirmButton.emit();
  }

  onClickCancel(){
    this.onClickCancelButton.emit();
  }
}
