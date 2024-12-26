import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SelectWindowComponent } from './select-window.component';

describe('SelectWindowComponent', () => {
  let component: SelectWindowComponent;
  let fixture: ComponentFixture<SelectWindowComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [SelectWindowComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SelectWindowComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
