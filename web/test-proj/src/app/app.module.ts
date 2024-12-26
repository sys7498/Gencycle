import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { SocketIoModule, SocketIoConfig } from 'ngx-socket-io';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { AppComponent } from './app.component';
import { LoadingComponent } from './component/loading/loading.component';
import { SelectWindowComponent } from './component/select-window/select-window.component';
import { ImagesComponent } from './component/images/images.component';
import { ChipComponent } from './component/chip/chip.component';

const config: SocketIoConfig = {
  url: '/',
  options: {
    path: '/socket.io/',
    transports: ['websocket'],
    upgrade: false,
  },
};

@NgModule({
  declarations: [AppComponent, LoadingComponent, SelectWindowComponent, ImagesComponent, ChipComponent],
  imports: [
    BrowserModule,
    SocketIoModule.forRoot(config),
    HttpClientModule,
    FormsModule,
  ],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
